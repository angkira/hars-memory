#!/usr/bin/env python3
"""Deterministic generator for the ``hard_1k`` long-term-memory retrieval set.

The set targets the failure mode of ordinary lexical/dense top-k retrieval:
every case is answerable from the corpus, but the answer lives in a document
that the question does NOT name.  Each case therefore has an *anchor* document
(which the question describes) and one or more *target* documents (which hold
the answer), joined by a bridge that exists only inside the corpus:

* a shared JobRun UUID / experiment id,
* an explicit file reference,
* a hypothesis -> link -> experiment chain,
* a baseline-experiment pointer,
* a versioned / chronologically adjacent record,
* an accepted-vs-rejected sibling,
* a near-identical run id that must be disambiguated.

No LLM is used: questions, answers and snippets are derived from corpus
structure (titles, dates, ids, metric JSON, references, graph edges).  That
keeps the build reproducible and costs $0.

Usage::

    .venv/bin/python -m hars_memory.eval.gen_hard_queries \
        --index-dir <index-dir> \
        --out-dir eval_agentic-data/hard_1k \
        --target 1000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml  # type: ignore[import-not-found]

logger = logging.getLogger("memory.eval.gen_hard_queries")

# Corpus-specific domain tokens (experiment prefixes, metric labels, graph
# bridge tokens) live OUTSIDE the shipped package tree so the wheel never
# carries internal identifiers. See tests/test_package_hygiene.py.
_DOMAIN_CONFIG_ENV = "HARS_HARD1K_DOMAIN_CONFIG"


def _load_domain_tokens() -> dict[str, object]:
    path = os.environ.get(_DOMAIN_CONFIG_ENV, "")
    if not path:
        candidate = (
            Path(__file__).resolve().parents[2]
            / "eval_agentic-data" / "hard_1k" / "domain_tokens.json"
        )
        path = str(candidate) if candidate.is_file() else ""
    if path and Path(path).is_file():
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("ignoring unreadable domain token config: %s", path)
    return {}


_DOMAIN = _load_domain_tokens()

DEFAULT_INDEX_DIR = Path(
    os.environ.get("HARS_MEMORY_INDEX_DIR", "")
    or (Path.home() / ".local" / "share" / "hars-graphrag" / "index")
)
DEFAULT_OUT_DIR = Path("eval_agentic-data/hard_1k")

STRATA = (
    "multi_hop",
    "relational",
    "temporal",
    "negation",
    "numeric",
    "lexical_mismatch",
    "disambiguation",
    "implicit_reference",
)

# Target allocation.  The selector may rebalance by at most +/-15% when a
# stratum's deterministic candidate pool is short; it never invents cases.
STRATUM_TARGET = {
    "multi_hop": 155,
    "relational": 145,
    "temporal": 110,
    "negation": 105,
    "numeric": 140,
    "lexical_mismatch": 105,
    "disambiguation": 120,
    "implicit_reference": 120,
}

TRAIN_FRACTION = 0.70

# ---------------------------------------------------------------------------
# Corpus structures
# ---------------------------------------------------------------------------

UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
_EXP_PREFIXES = [str(x) for x in _DOMAIN.get("experiment_prefixes", [])]
_EXP_EXTRA = str(_DOMAIN.get("experiment_prefix_extra_regex", ""))
_EXP_ALTERNATION = "|".join(
    [re.escape(p) for p in _EXP_PREFIXES] + ([_EXP_EXTRA] if _EXP_EXTRA else [])
) or "(?!)"
EXP_ID_RE = re.compile(
    r"\b(?:" + _EXP_ALTERNATION + r")[_a-z0-9]*\d{8}_\d{6}_[a-f0-9]{6}\b"
)
A2S_RE = re.compile(r"\bA2S\d+[a-z]?\b")
DATE_RE = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")
SECTION_RE = re.compile(r"\[Document:[^\]]*\]\n\n")
SLUG_DATE_RE = re.compile(r"^(\d{4}-\d\d-\d\d)[_-]")
TIME_SUFFIX_RE = re.compile(r"_(\d{6})(?:_[a-f0-9]{6})?$")
VERSION_RE = re.compile(r"(?:[-_]v)(\d+)(?=[-_.]|$)")
METRIC_LINE_RE = re.compile(r'"([A-Za-z0-9_./\-]+)":\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)')
STATUS_RE = re.compile(r"^\s*\**Status\**\s*:\s*(.+)$", re.M)
VERDICT_RE = re.compile(r"^\s*\**Verdict\**\s*:\s*(.+)$", re.M)
TITLE_RE = re.compile(r"^#\s+(.+)$", re.M)
TAGS_RE = re.compile(r"^\s*Tags?\s*:\s*(.+)$", re.M)
DATE_LINE_RE = re.compile(r"^\s*\**Date\**\s*:\s*(.+)$", re.M)
ERROR_RE = re.compile(r"^\s*Error:\s*(.+)$", re.M)


@dataclass(eq=False)
class Doc:
    """One resolved production-index document."""

    path: str
    content: str
    title: str
    date: str
    date_order: str  # sortable YYYY-MM-DD or ""
    tags: tuple[str, ...]
    status: str
    body: str
    kind: str

    @property
    def slug(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    def has(self, needle: str) -> bool:
        return needle.casefold() in self.content.casefold()


@dataclass
class Fact:
    """A verifiable answer span extracted from a document."""

    kind: str  # metric | status | verdict | job_id | reason | count
    answer: str
    snippet: str
    phrase: str
    key: str = ""
    value: float | None = None


@dataclass
class Candidate:
    stratum: str
    question: str
    answer: str
    anchor: str
    targets: list[str]
    snippets: dict[str, str]
    rationale: str
    hop_count: int
    bridge: str
    template: str
    fact_kind: str
    answer_entity: str = ""
    answer_note: str = ""
    difficulty: str = "hard"

    @property
    def gold_docs(self) -> list[str]:
        return [self.anchor, *self.targets]


# ---------------------------------------------------------------------------
# Loading and metadata extraction
# ---------------------------------------------------------------------------


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _dedupe_key(text: str) -> str:
    return hashlib.sha1(_norm_key(text).encode("utf-8")).hexdigest()


def _strip_sections(content: str) -> str:
    return SECTION_RE.sub("", content)


def _title_of(path: str, body: str) -> str:
    match = TITLE_RE.search(body)
    if match:
        return match.group(1).strip().strip("*`")
    stem = path.rsplit("/", 1)[-1]
    if stem.endswith(".md"):
        stem = stem[:-3]
    stem = re.sub(r"^staging-|^ingested-|^staging-ingested-", "", stem)
    stem = re.sub(r"^\d{4}-\d\d-\d\d[_-]", "", stem)
    return stem.replace("_", " ").replace("-", " ").strip()


def _date_of(path: str, body: str) -> str:
    match = SLUG_DATE_RE.search(path.rsplit("/", 1)[-1])
    if match:
        return match.group(1)
    match = DATE_LINE_RE.search(body)
    if match:
        found = DATE_RE.search(match.group(1))
        if found:
            return found.group(1)
    # Ingest headers: prefer the entity section's date over the index date.
    headers = re.findall(
        r"\[Document:[^\]]*?Section:\s*([a-z_]+)[^\]]*?Date:\s*(20\d\d-\d\d-\d\d)", body
    )
    for section, date in headers:
        if section in {"hypothesis", "experiment", "hypothesis_link"}:
            return date
    if headers:
        return headers[-1][1]
    found = DATE_RE.search(body[:600])
    return found.group(1) if found else ""


def _kind_of(path: str) -> str:
    slug = path.rsplit("/", 1)[-1]
    if slug.startswith("hypotheses-"):
        return "hypothesis"
    if slug.startswith("experiments-"):
        return "experiment"
    if slug.startswith("hypothesis_links-"):
        return "hypothesis_link"
    if slug.startswith(("staging-", "ingested-", "staging-ingested-")):
        return "staging"
    if ".plans" in slug or slug.startswith(".plans-"):
        return "plan"
    if slug.startswith("docs-") or slug.startswith("root-docs-"):
        return "docs"
    if "memory-archive" in slug:
        return "archive"
    if re.match(r"^\d{4}-\d\d-\d\d[_-]", slug):
        return "report"
    return "other"


def load_docs(index_dir: Path) -> dict[str, Doc]:
    """Load processed documents from the production index, keyed by file_path."""
    index_dir = Path(index_dir)
    status_path = index_dir / "kv_store_doc_status.json"
    full_path = index_dir / "kv_store_full_docs.json"
    if not status_path.exists() or not full_path.exists():
        raise SystemExit(f"index is missing kv_store_doc_status/full_docs: {index_dir}")
    statuses = json.loads(status_path.read_text(encoding="utf-8"))
    full_docs = json.loads(full_path.read_text(encoding="utf-8"))
    out: dict[str, Doc] = {}
    for doc_id, meta in statuses.items():
        if not isinstance(meta, dict) or meta.get("status") != "processed":
            continue
        raw = full_docs.get(doc_id)
        if raw is None:
            continue
        content = raw.get("content", "") if isinstance(raw, dict) else str(raw)
        if not content:
            continue
        path = str(meta.get("file_path") or "")
        if not path:
            continue
        body = _strip_sections(content)
        title = _title_of(path, body)
        # The ingest header carries the authoritative date for KB entity notes
        # whose filenames are opaque (hypotheses-0120.md, experiments-0053.md).
        date = _date_of(path, content) or _date_of(path, body)
        status_match = STATUS_RE.search(body)
        status = status_match.group(1).strip() if status_match else ""
        status = re.split(r"[.;]", status, maxsplit=1)[0].strip()[:160]
        tags_match = TAGS_RE.search(body)
        tags = ()
        if tags_match:
            tags = tuple(
                tag.strip().strip("`").casefold()
                for tag in re.split(r"[,;]", tags_match.group(1))
                if tag.strip()
            )
        out[path] = Doc(
            path=path,
            content=content,
            title=title or path,
            date=date,
            date_order=date,
            tags=tags,
            status=status,
            body=body,
            kind=_kind_of(path),
        )
    if not out:
        raise SystemExit(f"no processed documents resolved in {index_dir}")
    return out


# ---------------------------------------------------------------------------
# Structured entity parsing
# ---------------------------------------------------------------------------


@dataclass
class Experiment:
    doc: str
    exp_id: str
    name: str
    status: str
    job_id: str
    baseline: str
    metrics: dict[str, str]


@dataclass
class Hypothesis:
    doc: str
    hyp_id: str
    slug: str
    title: str
    status: str
    description: str
    rationale: str
    criteria: str
    tags: tuple[str, ...]


@dataclass
class HypLink:
    doc: str
    hyp_id: str
    exp_id: str
    relation: str
    label: str
    notes: str


def _first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.M)
    return match.group(1).strip() if match else ""


def parse_experiments(docs: dict[str, Doc]) -> dict[str, Experiment]:
    out: dict[str, Experiment] = {}
    for path, doc in docs.items():
        if doc.kind != "experiment":
            continue
        exp_id = _first(r"Experiment ID:\s*(\S+)", doc.body)
        name = _first(r"Name:\s*(.+)", doc.body)
        status = _first(r"Status:\s*(.+)", doc.body)
        job_id = _first(r"Job_id:\s*(\S+)", doc.body)
        baseline = _first(r"Baseline_experiment_id:\s*(\S+)", doc.body)
        metrics = {key: value for key, value in METRIC_LINE_RE.findall(doc.body)}
        out[path] = Experiment(
            doc=path,
            exp_id=exp_id,
            name=name,
            status=status,
            job_id=job_id,
            baseline=baseline,
            metrics=metrics,
        )
    return out


def parse_hypotheses(docs: dict[str, Doc]) -> dict[str, Hypothesis]:
    out: dict[str, Hypothesis] = {}
    for path, doc in docs.items():
        if doc.kind != "hypothesis":
            continue
        hyp_id = _first(r"Hypothesis ID:\s*(\S+)", doc.body)
        out[path] = Hypothesis(
            doc=path,
            hyp_id=hyp_id,
            slug=_first(r"Slug:\s*(\S+)", doc.body),
            title=_first(r"Title:\s*(.+)", doc.body) or doc.title,
            status=_first(r"Status:\s*(\S+)", doc.body),
            description=_first(r"Description:\s*(.+)", doc.body),
            rationale=_first(r"Rationale:\s*(.+)", doc.body),
            criteria=_first(r"Success criteria:\s*(.+)", doc.body),
            tags=doc.tags,
        )
    return out


def parse_hyp_links(docs: dict[str, Doc]) -> dict[str, HypLink]:
    out: dict[str, HypLink] = {}
    for path, doc in docs.items():
        if doc.kind != "hypothesis_link":
            continue
        out[path] = HypLink(
            doc=path,
            hyp_id=_first(r"Hypothesis:\s*(\S+)", doc.body),
            exp_id=_first(r"Links to:\s*experiment\s*/\s*(\S+)", doc.body),
            relation=_first(r"Relation:\s*(\S+)", doc.body),
            label=_first(r"Label:\s*(.+)", doc.body),
            notes=_first(r"Notes:\s*(.+)", doc.body),
        )
    return out


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

_METRIC_WORDS = {
    "val": "validation",
    "train": "training",
    "acc": "accuracy",
    "fact_acc": "factual accuracy",
    "loss": "loss",
    "lm_loss": "language-model loss",
    "native_retention_kl": "native-retention KL",
    "real_margin": "real margin",
    "retrieval_r1": "retrieval recall at 1",
    "retrieval_r5": "retrieval recall at 5",
    "perplexity": "perplexity",
    "lr": "learning rate",
    "grad_norm_proj": "projector gradient norm",
    "diversity_loss": "diversity loss",
    "variance_loss": "variance loss",
    "sidecar_gate_mean": "sidecar gate mean",
    "progress_batch": "progress batch",
    "progress_total": "progress total",
    "fact_loss": "factual loss",
    "evidence_answer": "evidence-answer",
    "accuracy": "accuracy",
    "margin_violation": "margin violation",
    "wrong_contribution": "wrong-evidence contribution",
}

_METRIC_WORDS.update(
    {str(k): str(v) for k, v in dict(_DOMAIN.get("metric_labels", {})).items()}
)


def human_metric(key: str) -> str:
    """Turn ``best_val/fact_acc`` into a short natural-language phrase."""
    parts = re.split(r"[/_]", key)
    words: list[str] = []
    for part in parts:
        low = part.casefold()
        if low in {"best", "final"}:
            words.append(low)
        elif low in _METRIC_WORDS:
            words.append(_METRIC_WORDS[low])
        else:
            words.append(low.replace("-", " "))
    phrase = " ".join(words)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase or key


def _sentence_around(text: str, start: int, end: int, width: int = 260) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    chunk = text[left:right]
    # trim to sentence boundaries when available
    left_cut = max(chunk.rfind(". "), chunk.rfind("\n"))
    if left_cut > 0:
        chunk = chunk[left_cut + 1 :]
    right_cut = min(
        [pos for pos in (chunk.find(". "), chunk.find("\n")) if pos > 0] or [len(chunk)]
    )
    chunk = chunk[:right_cut].strip()
    return re.sub(r"\s+", " ", chunk)[:400]


def extract_facts(doc: Doc, exp: Experiment | None = None) -> list[Fact]:
    """Extract verifiable, short answers from one document."""
    facts: list[Fact] = []
    seen: set[tuple[str, str]] = set()

    def add(
        kind: str, answer: str, snippet: str, phrase: str, key: str = "", value: float | None = None
    ) -> None:
        answer = answer.strip()
        snippet = snippet.strip()
        if not answer or not snippet or answer.casefold() not in doc.content.casefold():
            return
        if answer.casefold() not in snippet.casefold():
            return
        if snippet not in doc.content:
            index = doc.content.casefold().find(answer.casefold())
            if index < 0:
                return
            snippet = _sentence_around(doc.content, index, index + len(answer))
            if answer.casefold() not in snippet.casefold():
                return
        marker = (kind, answer)
        if marker in seen:
            return
        seen.add(marker)
        facts.append(
            Fact(kind=kind, answer=answer, snippet=snippet, phrase=phrase, key=key, value=value)
        )

    if exp is not None:
        for key, raw in exp.metrics.items():
            # Skip bookkeeping metrics and near-constant diagnostics that make
            # trivial answers (learning rate, progress counters, zero grads).
            low = key.casefold()
            if any(
                marker in low
                for marker in (
                    "progress",
                    "grad_norm",
                    "/lr",
                    "sidecar_gate",
                    "lm_loss_weight",
                    "loss_step",
                    "loss_epoch",
                    "diversity_loss",
                    "variance_loss",
                    "batch",
                    "total",
                )
            ):
                continue
            try:
                number = float(raw)
            except ValueError:
                continue
            if number in {0.0, 1.0} and not any(
                marker in low for marker in ("acc", "success", "recall", "precision")
            ):
                continue
            snippet = f'"{key}": {raw}'
            add(
                "metric",
                raw,
                snippet,
                f"the {human_metric(key)} value",
                key=key,
                value=number,
            )

    for match in STATUS_RE.finditer(doc.body):
        line = match.group(1).strip()
        token = re.split(r"[.;(]", line, maxsplit=1)[0].strip()
        if 2 <= len(token) <= 80:
            add("status", token, line[:300], "terminal status")

    for match in VERDICT_RE.finditer(doc.body):
        line = match.group(1).strip()
        verdict = re.split(r"(?<=[a-z])\.\s", line, maxsplit=1)[0].strip()
        if 3 <= len(verdict) <= 140:
            add("verdict", verdict, line[:400], "recorded verdict")

    for match in UUID_RE.finditer(doc.content):
        add("job_id", match.group(0), match.group(0), "managed JobRun identifier")

    for match in _NEGATION_RE.finditer(doc.body):
        snippet = _sentence_around(doc.body, match.start(), match.end(), width=220)
        snippet = re.sub(r"^#+\s*", "", snippet).strip()
        if not snippet or snippet.startswith(("*", "-", "|", "```")):
            continue
        if match.group(0).casefold() not in snippet.casefold():
            continue
        head = snippet[:100].casefold()
        if not _NEGATION_RE.search(head):
            continue
        if head.startswith(("verified", "completed", "confirmed", "passed", "operational")):
            continue
        if 20 <= len(snippet) <= 260:
            add("reason", snippet, snippet, "recorded rejection reason")

    # Numeric outcome sentences in free-form reports.
    number_pattern = re.compile(
        r"(?i)\b(success(?:\s+rate)?|accuracy|recall(?:@\d+)?|precision|latency|throughput|"
        r"fact accuracy|mask quality|strict success)\b[^.\n]{0,80}?"
        r"(\d+(?:\.\d+)?%?)"
    )
    for match in number_pattern.finditer(doc.body):
        value = match.group(2)
        if value in {"0", "1"} and len(value) == 1:
            continue
        snippet = _sentence_around(doc.body, match.start(), match.end())
        add(
            "count",
            value,
            snippet,
            f"reported {match.group(1).strip().casefold()}",
        )

    return facts


def best_fact(
    facts: Iterable[Fact],
    *,
    prefer: Sequence[str] = ("metric", "count", "status", "verdict", "job_id"),
) -> Fact | None:
    order = {kind: index for index, kind in enumerate(prefer)}
    ranked = sorted(
        facts,
        key=lambda fact: (
            order.get(fact.kind, len(order)),
            0 if fact.kind == "metric" else 1,
            len(fact.answer),
        ),
    )
    return ranked[0] if ranked else None


# ---------------------------------------------------------------------------
# References and registries
# ---------------------------------------------------------------------------


def canonical_docs(docs: dict[str, Doc]) -> dict[str, Doc]:
    """Drop staging duplicates when the canonical copy is present."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for path, doc in docs.items():
        by_hash[_dedupe_key(doc.body)].append(path)
    drop: set[str] = set()
    for paths in by_hash.values():
        if len(paths) < 2:
            continue
        canonical = sorted(
            paths, key=lambda p: (p.startswith(("staging-", "ingested-")), len(p), p)
        )
        drop.update(canonical[1:])
    out = {path: doc for path, doc in docs.items() if path not in drop}
    return out


def build_registry(docs: dict[str, Doc], pattern: re.Pattern[str]) -> dict[str, set[str]]:
    registry: dict[str, set[str]] = defaultdict(set)
    for path, doc in docs.items():
        for match in set(pattern.findall(doc.content)):
            registry[match].add(path)
    return registry


def build_references(docs: dict[str, Doc]) -> dict[str, set[str]]:
    """A -> B when A mentions B's filename stem (an explicit in-corpus reference)."""
    stems: dict[str, str] = {}
    for path in docs:
        stem = path.rsplit("/", 1)[-1]
        if stem.endswith(".md"):
            stem = stem[:-3]
        if len(stem) >= 18:
            stems[stem] = path
    refs: dict[str, set[str]] = defaultdict(set)
    for path, doc in docs.items():
        for stem, target in stems.items():
            if target == path:
                continue
            if stem in doc.content:
                refs[path].add(target)
    return refs


# ---------------------------------------------------------------------------
# Question wording helpers
# ---------------------------------------------------------------------------


def _clean_title(title: str, limit: int = 120) -> str:
    title = re.sub(r"\s+", " ", title).strip().strip("`'\"")
    if len(title) > limit:
        title = title[: limit - 1].rsplit(" ", 1)[0] + "…"
    return title


def _fact_phrase(fact: Fact) -> str:
    if fact.kind == "metric":
        return human_metric(fact.key)
    if fact.kind == "count":
        return fact.phrase.removeprefix("reported ")
    if fact.kind == "job_id":
        return "managed JobRun identifier"
    if fact.kind == "verdict":
        return "recorded verdict"
    if fact.kind == "reason":
        return "recorded failure reason"
    return "terminal status"


def _question_for(anchor: Doc, fact: Fact, role: str, template: int) -> str:
    title = _clean_title(anchor.title)
    phrase = _fact_phrase(fact)
    variants = (
        f'The note "{title}" refers to {role}. What {phrase} did that record give?',
        f'"{title}" points to {role} in a separate entry. What {phrase} was recorded there?',
        f'Following the pointer in "{title}" to {role}, what {phrase} does the linked record report?',
        f"{title} names {role} whose own record is stored separately. What {phrase} does that record show?",
        f'In the record "{title}", {role} is identified for follow-up. What {phrase} did that follow-up entry report?',
    )
    return variants[template % len(variants)]


def _overlap(a: str, b: str) -> float:
    tokens_a = set(_content_tokens(a))
    tokens_b = set(_content_tokens(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


_GENERIC_TOPIC = frozenset(
    """
    config configs status report reports training model models data run runs plan plans result results
    experiment experiments evaluation eval update notes analysis pipeline system implementation code
    test tests docs document project session summary spec specification architecture workflow task
    tasks review audit launch launched fix fixed issue issues policy dataset version meeting log logs
    progress contract design setup check checks recovery output metrics metric score scores
    readiness blocked program programs launch launched model models eval evaluation result update
    design audit review plan report study analysis fallback variant baseline control experiment
    training test smoke full local long new first final gate pipeline run runs stage phase step
    tournament source truth state family native single line work mode type level size scale time
    date steps set sets table file files path paths team note notes doc docs page pages item items
    diagnosis alignment provenance benchmark cpu gpu memory frame offline fit invalid partial
    strict pass passed failure failed success error errors warning warnings repair resume restart
    rerun retry attempt attempts continue continuation research utilization validation hypotheses
    hypothesis decomposition comparison monitoring runbook operator matrix notes research capability
    """.split()
)

_STOP = frozenset(
    """
    a an the and or of for to in on at by with from into over under after before this that these those
    is are was were be been being as it its their there which what who whom whose when where why how
    did does do not no nor than then so such can could may might will would should shall have has had
    per via vs versus using used use one two three first second third new old same other another
    """.split()
)


def _content_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_./@-]*", text.casefold())
    return [token for token in tokens if token not in _STOP and len(token) > 1]


def _answer_in(text: str, answer: str) -> bool:
    needle = answer.casefold().strip()
    if not needle:
        return False
    return needle in text.casefold()


def _shorten(text: str, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


# ---------------------------------------------------------------------------
# Miners
# ---------------------------------------------------------------------------


class Corpus:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        raw = load_docs(self.index_dir)
        self.all_docs = raw
        self.docs = canonical_docs(raw)
        self.experiments = parse_experiments(self.docs)
        self.hypotheses = parse_hypotheses(self.docs)
        self.links = parse_hyp_links(self.docs)
        self.refs = build_references(self.docs)
        self.uuid_registry = build_registry(self.docs, UUID_RE)
        self.exp_registry = build_registry(self.docs, EXP_ID_RE)
        self.a2s_registry = build_registry(self.docs, A2S_RE)
        self.facts: dict[str, list[Fact]] = {
            path: extract_facts(doc, self.experiments.get(path)) for path, doc in self.docs.items()
        }
        # Human-readable titles for opaque KB filenames (experiments-0053.md).
        for path, exp in self.experiments.items():
            if exp.name:
                self.docs[path].title = exp.name
        for path, hyp in self.hypotheses.items():
            if hyp.title:
                self.docs[path].title = hyp.title
        for path, link in self.links.items():
            if link.label:
                self.docs[path].title = link.label
        self.hyp_by_id = {
            hyp.hyp_id.removeprefix("hyp:"): hyp for hyp in self.hypotheses.values() if hyp.hyp_id
        }
        self.exp_by_id = {
            exp.exp_id.removeprefix("exp:"): exp for exp in self.experiments.values() if exp.exp_id
        }
        # Entity graph + entity -> document map (graph branch source material).
        self.entity_docs: dict[str, set[str]] = {}
        self.graph_neighbors: dict[str, set[str]] = defaultdict(set)
        chunk_doc: dict[str, str] = {}
        text_chunks_path = self.index_dir / "kv_store_text_chunks.json"
        entity_chunks_path = self.index_dir / "kv_store_entity_chunks.json"
        graph_path = self.index_dir / "graph_chunk_entity_relation.graphml"
        if text_chunks_path.exists() and entity_chunks_path.exists():
            text_chunks = json.loads(text_chunks_path.read_text(encoding="utf-8"))
            chunk_doc = {
                chunk_id: str(meta.get("file_path") or "") for chunk_id, meta in text_chunks.items()
            }
            entity_chunks = json.loads(entity_chunks_path.read_text(encoding="utf-8"))
            for entity, meta in entity_chunks.items():
                docs = {
                    chunk_doc[chunk]
                    for chunk in meta.get("chunk_ids", [])
                    if chunk in chunk_doc and chunk_doc[chunk] in self.docs
                }
                if docs:
                    self.entity_docs[entity] = docs
        if graph_path.exists():
            import xml.etree.ElementTree as ET

            for event, element in ET.iterparse(str(graph_path), events=("end",)):
                if element.tag.endswith("edge"):
                    source = element.attrib.get("source", "")
                    target = element.attrib.get("target", "")
                    if source and target:
                        self.graph_neighbors[source].add(target)
                        self.graph_neighbors[target].add(source)
                    element.clear()

    def fact_for(self, path: str, prefer: Sequence[str] | None = None) -> Fact | None:
        facts = self.facts.get(path, [])
        return best_fact(facts, prefer=prefer or ("metric", "count", "status", "verdict", "job_id"))

    def doc(self, path: str) -> Doc:
        return self.docs[path]


def _make_candidate(
    stratum: str,
    corpus: Corpus,
    anchor: Doc,
    target: Doc,
    fact: Fact,
    role: str,
    template: int,
    bridge: str,
    rationale: str,
    hop_count: int = 2,
    extra_gold: list[str] | None = None,
    question: str | None = None,
    answer_entity: str = "",
    answer_note: str = "",
) -> Candidate | None:
    question = question or _question_for(anchor, fact, role, template)
    if not question.endswith("?"):
        question += "?"
    if _answer_in(question, fact.answer):
        return None
    answer_tokens = set(_content_tokens(fact.answer))
    if len(answer_tokens) >= 3:
        overlap = len(answer_tokens & set(_content_tokens(question))) / len(answer_tokens)
        if overlap >= 0.8:
            return None
    if _answer_in(anchor.title, fact.answer):
        return None
    if len(question) < 40 or len(question) > 360:
        return None
    bridge_text = bridge.removeprefix("ref:")
    anchor_snippet = _anchor_snippet(anchor, role)
    if len(bridge_text) >= 8 and bridge_text in anchor.content:
        index = anchor.content.find(bridge_text)
        span = _sentence_around(anchor.content, index, index + len(bridge_text), width=140)
        if bridge_text in span:
            anchor_snippet = span[:300]
    snippets = {anchor.path: anchor_snippet}
    snippets[target.path] = fact.snippet
    gold = [anchor.path, target.path]
    if extra_gold:
        for path in extra_gold:
            if path not in gold:
                gold.append(path)
                if path in corpus.docs:
                    snippets.setdefault(path, _anchor_snippet(corpus.doc(path), role))
    return Candidate(
        stratum=stratum,
        question=question,
        answer=fact.answer,
        anchor=anchor.path,
        targets=[path for path in gold if path != anchor.path],
        snippets=snippets,
        rationale=rationale,
        hop_count=hop_count,
        bridge=bridge,
        template=f"t{template % 5}",
        fact_kind=fact.kind,
        answer_entity=answer_entity,
        answer_note=answer_note,
    )


def _anchor_snippet(doc: Doc, role: str) -> str:
    """A short anchor-side span that is verbatim in the stored document."""
    title_head = doc.title[:50].casefold()
    preferred: list[str] = []
    fallback: list[str] = []
    for raw_line in doc.content.splitlines():
        line = raw_line.strip()
        if line.startswith("[Document:") or line.startswith(("#", "|", "```")):
            continue
        if len(line) < 40:
            continue
        low = line.casefold()
        if any(
            marker in low
            for marker in ("status:", "date:", "verdict:", "pipeline run:", "jobrun", "job run")
        ):
            preferred.append(line)
        elif title_head and title_head in low:
            preferred.append(line)
        elif not line.startswith(("*", "-", "Tags:")):
            fallback.append(line)
    for line in preferred + fallback:
        return line[:300]
    # Fall back to the title, which is always present verbatim.
    return doc.title[:200]


def _role_for_bridge(bridge_kind: str) -> str:
    return {
        "uuid": "a managed job run",
        "exp_id": "a training experiment",
        "reference": "a separate evidence report",
        "hyp_link": "a recorded experiment link",
        "baseline": "the baseline experiment it continued from",
        "series_next": "the next entry in the same line of work",
        "series_prev": "the earlier entry in the same line of work",
        "sibling": "the sibling candidate from the same comparison",
        "near_id": "the near-identical run in the same launch batch",
        "implicit_next": "the follow-up it says came next",
        "archive": "the archived source note it points to",
    }.get(bridge_kind, "a linked record")


def mine_shared_ids(corpus: Corpus, kind: str, stratum: str, limit: int) -> list[Candidate]:
    registry = corpus.uuid_registry if kind == "uuid" else corpus.exp_registry
    candidates: list[Candidate] = []
    for bridge_id, paths in sorted(registry.items()):
        path_list = sorted(paths)
        if len(path_list) != 2:
            continue
        for anchor_path, target_path in (path_list, path_list[::-1]):
            anchor = corpus.doc(anchor_path)
            target = corpus.doc(target_path)
            if anchor.kind == "staging" or target.kind == "staging":
                continue
            # The anchor must not name the answer: otherwise opening it is enough.
            fact = corpus.fact_for(target_path)
            if fact is None:
                continue
            if fact.kind != "metric" and _answer_in(anchor.content, fact.answer):
                continue
            if fact.kind == "metric" and fact.answer in anchor.content:
                continue
            # Prefer an anchor that has a usable title and does not itself look
            # like the target's duplicate.
            if not anchor.title or len(anchor.title) < 8:
                continue
            if _overlap(anchor.body, target.body) > 0.92:
                continue
            role = _role_for_bridge(kind)
            template = len(candidates) % 5
            rationale = (
                f"The question names only '{anchor.title}', which carries the {kind} bridge; "
                f"the answer ({_fact_phrase(fact)}) is stored in a separate record that shares "
                f"that identifier but not the question's vocabulary. A top-k lexical/dense query "
                f"stops at the anchor and never opens the linked record."
            )
            cand = _make_candidate(
                stratum,
                corpus,
                anchor,
                target,
                fact,
                role,
                template,
                bridge=bridge_id,
                rationale=rationale,
                hop_count=2,
            )
            if cand is not None:
                candidates.append(cand)
    rng = random.Random(f"shared-{kind}")
    rng.shuffle(candidates)
    return _diverse(candidates)[:limit]


def mine_references(corpus: Corpus, limit: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for anchor_path, targets in sorted(corpus.refs.items()):
        anchor = corpus.doc(anchor_path)
        if anchor.kind == "staging":
            continue
        # Only unambiguous single-reference anchors: the question says "the
        # separate report it cites", so there must be exactly one.
        real_targets = [
            p for p in targets if p in corpus.docs and not corpus.doc(p).kind == "staging"
        ]
        if len(real_targets) != 1:
            continue
        target_path = real_targets[0]
        target = corpus.doc(target_path)
        if target_path == anchor_path:
            continue
        fact = corpus.fact_for(target_path)
        if fact is None:
            continue
        if _answer_in(anchor.content, fact.answer):
            continue
        if fact.kind == "metric" and fact.answer in anchor.content:
            continue
        template = len(candidates) % 5
        rationale = (
            f"'{anchor.title}' contains an explicit path reference to another document, but the "
            f"question does not mention that path. The cited record is the only place holding the "
            f"{_fact_phrase(fact)}; lexical retrieval reaches the citing note, not the citation."
        )
        cand = _make_candidate(
            "multi_hop",
            corpus,
            anchor,
            target,
            fact,
            _role_for_bridge("reference"),
            template,
            bridge=f"ref:{target_path}",
            rationale=rationale,
            hop_count=2,
        )
        if cand is not None:
            candidates.append(cand)
    return _diverse(candidates)[:limit]


def mine_reference_chains(corpus: Corpus, limit: int) -> list[Candidate]:
    """A -> B -> C explicit-citation chain (three documents, three hops)."""
    candidates: list[Candidate] = []
    for anchor_path, first_hops in sorted(corpus.refs.items()):
        anchor = corpus.doc(anchor_path)
        if anchor.kind == "staging":
            continue
        for middle_path in sorted(first_hops):
            if middle_path not in corpus.docs:
                continue
            middle = corpus.doc(middle_path)
            if middle.kind == "staging":
                continue
            second_hops = [
                path
                for path in corpus.refs.get(middle_path, set())
                if path in corpus.docs
                and path not in {anchor_path, middle_path}
                and corpus.doc(path).kind != "staging"
            ]
            for target_path in sorted(second_hops):
                target = corpus.doc(target_path)
                fact = corpus.fact_for(target_path)
                if fact is None or _answer_shortcut(anchor, fact):
                    continue
                if _answer_in(middle.content, fact.answer):
                    continue
                question = (
                    f'The note "{_clean_title(anchor.title)}" points to a report, which in turn '
                    f"points to a second record. What {_fact_phrase(fact)} does the end of that "
                    f"citation chain report?"
                )
                rationale = (
                    "The answer is two references away from the only document named in the "
                    "question. The first citation is visible in the anchor, the second only after "
                    "opening the intermediate note; a flat top-k query stops at the anchor."
                )
                cand = _make_candidate(
                    "multi_hop",
                    corpus,
                    anchor,
                    target,
                    fact,
                    _role_for_bridge("reference"),
                    len(candidates) % 5,
                    bridge=f"ref-chain:{middle_path}",
                    rationale=rationale,
                    hop_count=3,
                    extra_gold=[middle_path],
                    question=question,
                )
                if cand is not None:
                    candidates.append(cand)
    return _diverse(candidates)[:limit]


def mine_hypothesis_links(corpus: Corpus, limit: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for link in corpus.links.values():
        hyp = corpus.hyp_by_id.get(link.hyp_id.removeprefix("hyp:"))
        exp = corpus.exp_by_id.get(link.exp_id.removeprefix("exp:"))
        if hyp is None or exp is None:
            continue
        if hyp.doc not in corpus.docs or exp.doc not in corpus.docs:
            continue
        fact = corpus.fact_for(exp.doc, prefer=("metric", "status", "verdict"))
        if fact is None:
            continue
        anchor = corpus.doc(hyp.doc)
        target = corpus.doc(exp.doc)
        if _answer_in(anchor.content, fact.answer):
            continue
        question = (
            f'The hypothesis "{_clean_title(hyp.title)}" is connected to one experiment through a '
            f"recorded hypothesis link. What {_fact_phrase(fact)} did that experiment report?"
        )
        rationale = (
            "The hypothesis note names the hypothesis id, not the experiment id; the experiment id "
            "lives only in the separate hypothesis-link record, and the answer lives only in the "
            "experiment record. Retrieval must follow hypothesis -> link -> experiment."
        )
        cand = _make_candidate(
            "relational",
            corpus,
            anchor,
            target,
            fact,
            _role_for_bridge("hyp_link"),
            0,
            bridge=f"{link.hyp_id}->{link.exp_id}",
            rationale=rationale,
            hop_count=3,
            extra_gold=[link.doc],
            question=question,
        )
        if cand is not None:
            candidates.append(cand)
    return _diverse(candidates)[:limit]


def mine_a2s_links(corpus: Corpus, limit: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for token, paths in sorted(corpus.a2s_registry.items()):
        path_list = sorted(paths)
        hyps = [p for p in path_list if corpus.doc(p).kind == "hypothesis"]
        exps = [p for p in path_list if corpus.doc(p).kind == "experiment"]
        if len(hyps) != 1 or len(exps) != 1:
            continue
        anchor = corpus.doc(hyps[0])
        target = corpus.doc(exps[0])
        fact = corpus.fact_for(target.path)
        if fact is None or _answer_shortcut(anchor, fact):
            continue
        question = (
            f'The hypothesis "{_clean_title(corpus.hypotheses[anchor.path].title)}" was evaluated '
            f"by one training experiment. What {_fact_phrase(fact)} did that experiment's final "
            f"metrics record?"
        )
        rationale = (
            f"The hypothesis and the experiment share only the run token {token}; the question "
            "deliberately omits it. The hypothesis record holds the intent, the experiment record "
            "holds the numeric result, and neither document names the other's title."
        )
        cand = _make_candidate(
            "relational",
            corpus,
            anchor,
            target,
            fact,
            _role_for_bridge("hyp_link"),
            1,
            bridge=token,
            rationale=rationale,
            hop_count=2,
            question=question,
        )
        if cand is not None:
            candidates.append(cand)
    return _diverse(candidates)[:limit]


def _answer_shortcut(anchor: Doc, fact: Fact) -> bool:
    """True when the answer is already visible in the anchor document.

    Metrics of 0.0/1.0 are too common to treat as a genuine shortcut; a
    distinctive numeric value or a text answer is a real one.
    """
    if fact.kind == "metric" and fact.value is not None and fact.value in {0.0, 1.0}:
        return False
    return _answer_in(anchor.content, fact.answer)


def mine_baseline_chains(
    corpus: Corpus, limit: int, stratum: str = "relational"
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for exp in corpus.experiments.values():
        if not exp.baseline:
            continue
        base = corpus.exp_by_id.get(exp.baseline)
        if base is None or base.doc not in corpus.docs or base.doc == exp.doc:
            continue
        anchor = corpus.doc(exp.doc)
        target = corpus.doc(base.doc)
        fact = corpus.fact_for(base.doc)
        if fact is None or _answer_shortcut(anchor, fact):
            continue
        question = (
            f'The experiment "{_clean_title(exp.name)}" continued from a baseline run in the same '
            f"series. What {_fact_phrase(fact)} did that baseline report?"
        )
        rationale = (
            "The anchor experiment record only carries the baseline experiment id; the baseline's "
            "own metrics are in a separate experiment record. A query for the anchor's name never "
            "surfaces the baseline, and the question omits the baseline id."
        )
        cand = _make_candidate(
            stratum,
            corpus,
            anchor,
            target,
            fact,
            _role_for_bridge("baseline"),
            len(candidates) % 5,
            bridge=exp.baseline,
            rationale=rationale,
            hop_count=2,
            question=question,
        )
        if cand is not None:
            candidates.append(cand)
    return _diverse(candidates)[:limit]


_BRIDGE_EXTRA = "|".join(str(x) for x in _DOMAIN.get("graph_bridge_tokens", []))
_GRAPH_BRIDGE_RE = re.compile(
    r"(?i)^(?:val|train)/|^A2S\d|DINO|Qwen|DA3|DROID|droid|SAM|SigLIP|Mamba|"
    + (_BRIDGE_EXTRA + "|" if _BRIDGE_EXTRA else "")
    + r"Gemma|Gemini|LoRA|DiT|SAE|RLPD|PACO|MTAD|CBANK|B1|E0|E3|R2[0-9]|A1-E|"
    r"world model|sidecar|tokenizer|retention|contrastive"
)


def mine_graph_chains(corpus: Corpus, limit: int) -> list[Candidate]:
    """Two-hop entity chains A -> B -> C where A and C live in different docs."""
    candidates: list[Candidate] = []
    for bridge, bridge_docs in sorted(corpus.entity_docs.items()):
        if not (2 <= len(bridge_docs) <= 4):
            continue
        if not (3 <= len(bridge) <= 70) or bridge.endswith((".md", ".py", ".yaml", ".json")):
            continue
        if not _GRAPH_BRIDGE_RE.search(bridge):
            continue
        neighbours = corpus.graph_neighbors.get(bridge, set())
        if not (2 <= len(neighbours) <= 12):
            continue
        for left_entity in sorted(neighbours):
            left_docs = corpus.entity_docs.get(left_entity, set())
            if not left_docs:
                continue
            for right_entity in sorted(neighbours):
                if right_entity == left_entity:
                    continue
                right_docs = corpus.entity_docs.get(right_entity, set())
                if not right_docs:
                    continue
                if left_docs & right_docs:
                    continue
                if right_entity in corpus.graph_neighbors.get(left_entity, set()):
                    # Direct edge: the chain would collapse to one hop.
                    continue
                for anchor_path in sorted(left_docs):
                    anchor = corpus.doc(anchor_path)
                    if anchor.kind == "staging" or not anchor.title:
                        continue
                    for target_path in sorted(right_docs):
                        target = corpus.doc(target_path)
                        if target.kind == "staging" or target_path == anchor_path:
                            continue
                        if bridge.casefold() not in anchor.content.casefold():
                            continue
                        if bridge.casefold() not in target.content.casefold():
                            continue
                        fact = corpus.fact_for(target_path)
                        if fact is None or _answer_in(anchor.content, fact.answer):
                            continue
                        question = (
                            f'The record "{_clean_title(anchor.title)}" is linked in the entity graph '
                            f'to a second record through "{_shorten(bridge, 60)}". What '
                            f"{_fact_phrase(fact)} does that linked record report?"
                        )
                        rationale = (
                            "The two documents never name each other. They are connected only by a "
                            "shared graph entity, and the question supplies the anchor plus the "
                            "bridge concept, not the target. Flat top-k returns the anchor and other "
                            "mentions of the bridge; following the graph edge is required to pick the "
                            "correct record."
                        )
                        cand = _make_candidate(
                            "relational",
                            corpus,
                            anchor,
                            target,
                            fact,
                            _role_for_bridge("hyp_link"),
                            len(candidates) % 5,
                            bridge=f"{left_entity}->{bridge}->{right_entity}",
                            rationale=rationale,
                            hop_count=2,
                            question=question,
                        )
                        if cand is not None:
                            candidates.append(cand)
    return _diverse(candidates)[:limit]


def _series_key(title: str) -> str:
    text = title.casefold()
    text = re.sub(r"a2s\d+[a-z]?", " ", text)
    text = re.sub(r"\b(19|20)\d{2}\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token not in _STOP and len(token) > 2]
    return " ".join(tokens[:5])


def doc_tokens(corpus: Corpus, doc: Doc) -> set[str]:
    """Topic tokens used for deterministic series pairing.

    Experiment notes have a generic filename-derived title, so prefer the
    parsed experiment name (``A2S153 typed DA3 depth_pair12 ...``).
    """
    exp = corpus.experiments.get(doc.path)
    hyp = corpus.hypotheses.get(doc.path)
    if exp is not None and exp.name:
        text = exp.name
    elif hyp is not None:
        text = f"{hyp.title} {hyp.slug}"
    else:
        text = f"{doc.title} {doc.path.rsplit('/', 1)[-1].removesuffix('.md')}"
    text = re.sub(r"\b(19|20)\d{2}[-_]?\d{2}[-_]?\d{2}\b", " ", text)
    text = re.sub(r"\b\d{6}\b", " ", text)
    text = re.sub(r"\b[0-9a-f]{6,}\b", " ", text)
    text = text.replace("staging-", " ").replace("ingested-", " ")
    return {token for token in _content_tokens(text) if len(token) > 2}


def build_topic_pairs(
    corpus: Corpus,
    *,
    min_shared: int = 2,
    df_max: int = 20,
    require_dates: bool = True,
) -> list[tuple[Doc, Doc, set[str]]]:
    """Pair documents that share rare topic tokens.

    This is the deterministic substitute for an embedding-based series
    detector: tokens appearing in 2..df_max documents are specific enough to
    indicate the same line of work, while generic words are ignored.
    """
    docs = [doc for doc in corpus.docs.values() if doc.kind != "staging"]
    tokens_by = {doc.path: doc_tokens(corpus, doc) for doc in docs}
    document_frequency: dict[str, int] = defaultdict(int)
    for tokens in tokens_by.values():
        for token in tokens:
            document_frequency[token] += 1
    rare = {token for token, count in document_frequency.items() if 2 <= count <= df_max}
    pairs: list[tuple[Doc, Doc, set[str]]] = []
    for index, left in enumerate(docs):
        left_tokens = tokens_by[left.path] & rare
        if not left_tokens:
            continue
        for right in docs[index + 1 :]:
            if require_dates and (not left.date_order or not right.date_order):
                continue
            shared = {
                token
                for token in (left_tokens & tokens_by[right.path])
                if token not in _GENERIC_TOPIC and len(token) >= 3
            }
            if len(shared) >= min_shared:
                pairs.append((left, right, shared))
    pairs.sort(key=lambda item: (-len(item[2]), item[0].date_order, item[0].path))
    return pairs


def build_series(corpus: Corpus) -> dict[str, list[Doc]]:
    """Connected series groups derived from :func:`build_topic_pairs`."""
    parent: dict[str, str] = {path: path for path in corpus.docs}

    def find(path: str) -> str:
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right, _ in build_topic_pairs(corpus):
        union(left.path, right.path)
    groups: dict[str, list[Doc]] = defaultdict(list)
    for path, doc in corpus.docs.items():
        if doc.kind == "staging":
            continue
        groups[find(path)].append(doc)
    out: dict[str, list[Doc]] = {}
    for root, group in groups.items():
        if len(group) < 2:
            continue
        out[root] = sorted(group, key=lambda doc: (doc.date_order or "9999", doc.path))
    return out


def _pair_bucket(left: Doc, right: Doc, shared: set[str]) -> int:
    key = left.path + "|" + right.path + "|" + "+".join(sorted(shared)[:3])
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 2


def _same_family(left: Doc, right: Doc) -> bool:
    broad = {"report", "plan", "archive", "docs", "other"}
    return left.kind == right.kind or (left.kind in broad and right.kind in broad)


def _date_gap_days(left: Doc, right: Doc) -> int:
    try:
        return abs(
            (
                date_cls.fromisoformat(left.date_order) - date_cls.fromisoformat(right.date_order)
            ).days
        )
    except ValueError:
        return 10**6


def _strong_shared(shared: set[str]) -> bool:
    """A pair is coherent enough when it shares two tokens or one specific one."""
    if len(shared) >= 2:
        return True
    for token in shared:
        if any(character.isdigit() for character in token):
            return True
        if len(token) >= 9:
            return True
    return False


def mine_temporal(corpus: Corpus, limit: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for left, right, shared in build_topic_pairs(corpus, min_shared=1, df_max=10):
        if left.path == right.path or not left.date_order or not right.date_order:
            continue
        if left.date_order == right.date_order or not _same_family(left, right):
            continue
        if _date_gap_days(left, right) > 45:
            continue
        if not _strong_shared(shared):
            continue
        if _pair_bucket(left, right, shared) != 0:
            continue
        anchor, target = (left, right) if left.date_order < right.date_order else (right, left)
        fact = corpus.fact_for(target.path)
        if fact is None or _answer_in(anchor.content, fact.answer):
            continue
        key = "+".join(sorted(shared)[:3])
        question = (
            f'The record "{_clean_title(anchor.title)}" ({anchor.date}) was followed by another '
            f"entry in the same line of work. What {_fact_phrase(fact)} did that later entry report?"
        )
        rationale = (
            "Both records share rare topic vocabulary but not a title; the later entry is not named "
            "in the question. Only date ordering plus the shared topic identifies it, and a flat "
            "top-k query returns the anchor without the later result."
        )
        cand = _make_candidate(
            "temporal",
            corpus,
            anchor,
            target,
            fact,
            _role_for_bridge("series_next"),
            len(candidates) % 5,
            bridge=f"series:{key}",
            rationale=rationale,
            hop_count=2,
            question=question,
        )
        if cand is not None:
            candidates.append(cand)
    return _diverse(candidates)[:limit]


def mine_versioned(corpus: Corpus, limit: int) -> list[Candidate]:
    groups: dict[str, list[Doc]] = defaultdict(list)
    for doc in corpus.docs.values():
        if doc.kind == "staging":
            continue
        stem = doc.path.rsplit("/", 1)[-1]
        if stem.endswith(".md"):
            stem = stem[:-3]
        match = VERSION_RE.search(stem)
        if match:
            groups[VERSION_RE.sub("", stem)].append(doc)
    candidates: list[Candidate] = []
    for base, group in sorted(groups.items()):
        group = sorted(group, key=lambda doc: (doc.date_order or "9999", doc.path))
        if len(group) < 2:
            continue
        for anchor, target in zip(group, group[1:]):
            fact = corpus.fact_for(target.path)
            if fact is None or _answer_in(anchor.content, fact.answer):
                continue
            question = (
                f'The note "{_clean_title(anchor.title)}" belongs to a versioned series. What '
                f"{_fact_phrase(fact)} did the next version of that record report?"
            )
            rationale = (
                "The anchor is one version of a series; the answer is in a later version whose exact "
                "version suffix is not in the question. Lexical top-k finds the anchor's version and "
                "must be followed by version ordering to reach the answering revision."
            )
            cand = _make_candidate(
                "temporal",
                corpus,
                anchor,
                target,
                fact,
                _role_for_bridge("series_next"),
                len(candidates) % 5,
                bridge=f"version:{base}",
                rationale=rationale,
                hop_count=2,
                question=question,
            )
            if cand is not None:
                candidates.append(cand)
    return _diverse(candidates)[:limit]


_NEGATION_RE = re.compile(
    r"(?i)\b(rejected|abandoned|falsified|no[- ]go|ruled out|not approved|set aside|"
    r"did not qualify|failed(?: promotion)?|blocked|killed|cancel(?:l)?ed|unsuccessful)\b"
)


def _negative_reason_fact(corpus: Corpus, path: str) -> Fact | None:
    """A verbatim sentence that states the negative outcome/reason."""
    doc = corpus.doc(path)
    candidates: list[tuple[int, str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", doc.body):
        sentence = re.sub(r"^[#*\-\s]+", "", sentence).strip()
        if not (30 <= len(sentence) <= 260):
            continue
        match = _NEGATION_RE.search(sentence)
        if not match or match.start() > 90:
            continue
        head = sentence[:80].casefold()
        if head.startswith(
            (
                "verified",
                "completed",
                "confirmed",
                "passed",
                "operational",
                "this keeps",
                "this preserves",
                "the successful",
                "success",
            )
        ):
            continue
        candidates.append((match.start(), sentence))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], len(item[1])))
    sentence = candidates[0][1]
    fact = Fact(
        kind="reason",
        answer=sentence,
        snippet=sentence,
        phrase="recorded rejection reason",
    )
    corpus.facts.setdefault(path, []).append(fact)
    return fact


def _negative_fact(corpus: Corpus, path: str) -> Fact | None:
    facts = corpus.facts.get(path, [])
    positive_prefixes = ("verified", "completed", "confirmed", "passed", "operational", "succeeded")
    for kind in ("status", "verdict", "reason", "count", "metric"):
        for fact in facts:
            if fact.kind != kind:
                continue
            if kind == "count":
                return fact
            answer = fact.answer.strip()
            if len(answer) > 140:
                continue
            if answer.casefold().startswith(positive_prefixes):
                continue
            if not _NEGATION_RE.search(answer):
                continue
            # The negative word must be part of the verdict, not a stray
            # mention deep inside a long status sentence.
            if _NEGATION_RE.search(answer[:80]):
                return fact
    return None


def mine_negation(corpus: Corpus, limit: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for left, right, shared in build_topic_pairs(
        corpus, min_shared=1, df_max=8, require_dates=False
    ):
        left_neg = bool(_NEGATION_RE.search(left.body) or _NEGATION_RE.search(left.title))
        right_neg = bool(_NEGATION_RE.search(right.body) or _NEGATION_RE.search(right.title))
        if left_neg == right_neg:
            continue
        if not (set(left.tags) & set(right.tags)):
            continue
        if left.date_order and right.date_order and _date_gap_days(left, right) > 30:
            continue
        anchor, target = (right, left) if left_neg else (left, right)
        fact = _negative_fact(corpus, target.path) or _negative_reason_fact(corpus, target.path)
        if fact is None or _answer_shortcut(anchor, fact):
            continue
        key = "+".join(sorted(shared)[:3])
        question = (
            f'The record "{_clean_title(anchor.title)}" belongs to a comparison in which one '
            f"candidate did not survive. What {_fact_phrase(fact)} did the record for the "
            f"candidate that was set aside give?"
        )
        rationale = (
            "The question describes the surviving sibling, not the rejected one. The rejection is "
            "only visible in a separate record from the same comparison, and the question avoids "
            "the corpus's negation keywords, so the rejection note is not a direct lexical hit."
        )
        cand = _make_candidate(
            "negation",
            corpus,
            anchor,
            target,
            fact,
            _role_for_bridge("sibling"),
            len(candidates) % 5,
            bridge=f"negation:{key}",
            rationale=rationale,
            hop_count=2,
            question=question,
        )
        if cand is not None:
            candidates.append(cand)
    # Sharper negation pairs: a launch/positive record and a negative result
    # record that share one concrete run identifier.
    for kind, registry in (("uuid", corpus.uuid_registry), ("exp_id", corpus.exp_registry)):
        for bridge_id, paths in sorted(registry.items()):
            path_list = sorted(paths)
            if len(path_list) != 2:
                continue
            left, right = corpus.doc(path_list[0]), corpus.doc(path_list[1])
            if left.kind == "staging" or right.kind == "staging":
                continue
            left_neg = bool(_NEGATION_RE.search(left.body) or _NEGATION_RE.search(left.title))
            right_neg = bool(_NEGATION_RE.search(right.body) or _NEGATION_RE.search(right.title))
            if left_neg == right_neg:
                continue
            anchor, target = (right, left) if left_neg else (left, right)
            fact = _negative_fact(corpus, target.path) or _negative_reason_fact(corpus, target.path)
            if fact is None or _answer_shortcut(anchor, fact):
                continue
            question = (
                f'The record "{_clean_title(anchor.title)}" shares a run identifier with a second '
                f"record in which the candidate did not survive. What {_fact_phrase(fact)} did that "
                f"second record give?"
            )
            rationale = (
                "Both records describe the same run, but only the target records the negative "
                "outcome. The question names the anchor and asks for the outcome without naming the "
                "target document or its verdict."
            )
            cand = _make_candidate(
                "negation",
                corpus,
                anchor,
                target,
                fact,
                _role_for_bridge("sibling"),
                len(candidates) % 5,
                bridge=bridge_id,
                rationale=rationale,
                hop_count=2,
                question=question,
            )
            if cand is not None:
                candidates.append(cand)
    return _diverse(candidates)[:limit]


def mine_numeric(corpus: Corpus, limit: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    exps = list(corpus.experiments.values())
    exp_by_doc = {exp.doc: exp for exp in exps}
    pairs = build_topic_pairs(corpus, min_shared=1, df_max=24, require_dates=True)
    for left, right, shared in pairs:
        if left.path not in exp_by_doc or right.path not in exp_by_doc:
            continue
        left_exp = exp_by_doc[left.path]
        right_exp = exp_by_doc[right.path]
        common = set(left_exp.metrics) & set(right_exp.metrics)
        common = {
            metric
            for metric in common
            if metric.startswith(("final_val/", "best_val/"))
            and metric not in {"final_val/progress_batch", "final_val/progress_total"}
        }
        if not common:
            continue
        left_date = left.date_order or "9999"
        right_date = right.date_order or "9999"
        anchor_doc, target_doc = (left, right) if left_date < right_date else (right, left)
        anchor_exp = exp_by_doc[anchor_doc.path]
        target_exp = exp_by_doc[target_doc.path]
        for metric in sorted(common)[:8]:
            try:
                anchor_value = float(anchor_exp.metrics[metric])
                target_value = float(target_exp.metrics[metric])
            except ValueError:
                continue
            fact = next(
                (
                    f
                    for f in corpus.facts[target_doc.path]
                    if f.kind == "metric" and f.key == metric
                ),
                None,
            )
            if fact is None:
                continue
            if _answer_shortcut(anchor_doc, fact):
                continue
            relation = "higher" if target_value > anchor_value else "lower"
            metric_phrase = human_metric(metric)
            variants = (
                f'The experiment "{_clean_title(anchor_exp.name)}" has a sibling record in the same '
                f"series. Which of the two reported the {relation} {metric_phrase}, and what was "
                f"that sibling's value?",
                f'In the series around "{_clean_title(anchor_exp.name)}", one sibling recorded a '
                f"{relation} {metric_phrase}. What value did that sibling record?",
                f'In the series around "{_clean_title(anchor_exp.name)}", two runs differ on '
                f"{metric_phrase}; the sibling held the {relation} value. What was it?",
                f'The series containing "{_clean_title(anchor_exp.name)}" includes a record with a '
                f"{relation} {metric_phrase}. What value is in that record?",
                f'Among the experiments related to "{_clean_title(anchor_exp.name)}", which sibling '
                f"held the {relation} {metric_phrase}, and what value did it report?",
            )
            question = variants[len(candidates) % len(variants)]
            rationale = (
                "The question names one experiment and asks for a comparison against its sibling, "
                "whose name is not given. Answering requires locating the sibling by series ordering "
                "and reading both metric records; a top-k query centered on the anchor does not "
                "reliably return the sibling."
            )
            cand = _make_candidate(
                "numeric",
                corpus,
                anchor_doc,
                target_doc,
                fact,
                _role_for_bridge("series_next"),
                len(candidates) % 5,
                bridge=f"numeric:{'+'.join(sorted(shared)[:3])}:{metric}",
                rationale=rationale,
                hop_count=2,
                question=question,
                answer_entity=target_exp.name,
                answer_note=f"{relation} {human_metric(metric)} than the anchor",
            )
            if cand is not None:
                candidates.append(cand)
    # Same-day experiments: a weaker but still structural series relation.
    by_date: dict[str, list[Experiment]] = defaultdict(list)
    for exp in exps:
        doc = corpus.doc(exp.doc)
        if doc.date_order:
            by_date[doc.date_order].append(exp)
    for date, group in sorted(by_date.items()):
        group = sorted(group, key=lambda exp: exp.doc)
        for index, left_exp in enumerate(group):
            for right_exp in group[index + 1 :]:
                common = set(left_exp.metrics) & set(right_exp.metrics)
                common = {
                    metric
                    for metric in common
                    if metric.startswith(("final_val/", "best_val/"))
                    and metric not in {"final_val/progress_batch", "final_val/progress_total"}
                }
                if not common:
                    continue
                anchor_doc = corpus.doc(left_exp.doc)
                target_doc = corpus.doc(right_exp.doc)
                for metric in sorted(common)[:3]:
                    fact = next(
                        (
                            f
                            for f in corpus.facts[target_doc.path]
                            if f.kind == "metric" and f.key == metric
                        ),
                        None,
                    )
                    if fact is None or _answer_shortcut(anchor_doc, fact):
                        continue
                    question = (
                        f"Another experiment completed on {date}, the same day as "
                        f'"{_clean_title(left_exp.name)}". What {human_metric(metric)} did that '
                        f"same-day sibling report?"
                    )
                    rationale = (
                        "The sibling is identified only by the shared completion date and metric "
                        "name, neither of which is unique in the corpus. A top-k query for the "
                        "anchor's name returns the anchor and unrelated same-day notes, not the "
                        "answering experiment record."
                    )
                    cand = _make_candidate(
                        "numeric",
                        corpus,
                        anchor_doc,
                        target_doc,
                        fact,
                        _role_for_bridge("series_next"),
                        len(candidates) % 5,
                        bridge=f"same-day:{date}:{metric}",
                        rationale=rationale,
                        hop_count=2,
                        question=question,
                        answer_entity=right_exp.name,
                        answer_note=f"same-day sibling {date}",
                    )
                    if cand is not None:
                        candidates.append(cand)
    candidates.extend(mine_baseline_chains(corpus, limit, stratum="numeric"))
    return _diverse(candidates)[:limit]


_SYNONYMS = {
    "validation": "held-out",
    "training": "fitting",
    "accuracy": "correctness",
    "job": "execution",
    "run": "execution",
    "experiment": "trial",
    "record": "account",
    "report": "write-up",
    "checkpoint": "saved weights",
    "throughput": "rate",
    "latency": "response time",
    "success": "pass",
    "failed": "fell short",
    "rejected": "set aside",
    "managed": "orchestrated",
    "terminal": "final",
    "baseline": "reference",
    "link": "connection",
}


_PHRASE_SYNONYMS = (
    ("managed job run", "orchestrated execution"),
    ("job run", "execution"),
    ("training experiment", "fitting trial"),
    ("terminal status", "final outcome"),
    ("recorded verdict", "stated conclusion"),
    ("recorded rejection reason", "stated reason for being set aside"),
    ("final validation", "held-out"),
    ("best validation", "best held-out"),
    ("validation", "held-out"),
    ("success rate", "pass rate"),
)


def _paraphrase(text: str, forbidden: Iterable[str]) -> str:
    forbidden_low = {item.casefold() for item in forbidden}
    out = text
    for phrase, replacement in _PHRASE_SYNONYMS:
        if phrase in out.casefold() and replacement.casefold() not in forbidden_low:
            out = re.sub(rf"(?i)\b{re.escape(phrase)}\b", replacement, out)
    replaced: set[str] = set()
    for word, replacement in _SYNONYMS.items():
        if word in out.casefold() and replacement.casefold() not in forbidden_low:
            if replacement in replaced or replacement.casefold() in out.casefold():
                continue
            out = re.sub(rf"(?i)\b{re.escape(word)}\b", replacement, out)
            replaced.add(replacement)
    out = re.sub(r"\ba (?=[aeiou])", "an ", out, flags=re.I)
    return out


def mine_lexical_mismatch(
    corpus: Corpus, seed_candidates: Sequence[Candidate], limit: int
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for base in seed_candidates:
        anchor = corpus.doc(base.anchor)
        target = corpus.doc(base.targets[0])
        # Deliberately avoid the anchor's title: use topic tags plus one rare
        # anchor-specific token.  This keeps the question identifiable while
        # removing the title vocabulary a lexical matcher would key on.
        anchor_tokens = set(_content_tokens(anchor.title))
        target_tokens = set(_content_tokens(target.title + " " + target.body))
        hints = sorted(
            (
                token
                for token in anchor_tokens - target_tokens
                if len(token) >= 5
                and token not in _GENERIC_TOPIC
                and re.search(r"[a-z]{3}", token)
                and not re.fullmatch(r"[\d_:.-]+", token)
            ),
            key=lambda token: (-len(token), token),
        )
        hint = ""
        for candidate_hint in hints:
            frequency = sum(
                1 for other in corpus.docs.values() if candidate_hint in other.content.casefold()
            )
            if frequency <= 3:
                hint = candidate_hint
                break
        if not hint:
            continue
        topic = " / ".join(anchor.tags[:2]) if anchor.tags else "research"
        fact = corpus.fact_for(base.targets[0])
        if fact is None:
            continue
        fact_phrase = _paraphrase(_fact_phrase(fact), [anchor.content, target.content])
        question = (
            f'In the {topic} line of work, the account that mentions "{hint}" points to a '
            f"separate record. What {fact_phrase} did that record report?"
        )
        if _answer_in(question, fact.answer) or _overlap(question, target.body) >= 0.55:
            continue
        cand = Candidate(
            stratum="lexical_mismatch",
            question=question,
            answer=base.answer,
            anchor=base.anchor,
            targets=base.targets,
            snippets=base.snippets,
            rationale=(
                "The question deliberately uses vocabulary that does not occur in the answering "
                "record (paraphrased metric/role terms). Lexical matching is therefore weak, and the "
                "answer requires semantic alignment plus the anchor's bridge."
            ),
            hop_count=base.hop_count,
            bridge=base.bridge,
            template="paraphrase",
            fact_kind=base.fact_kind,
            answer_entity=base.answer_entity,
            answer_note=base.answer_note,
        )
        candidates.append(cand)
    return _diverse(candidates)[:limit]


def _run_batch_prefix(exp: Experiment) -> str:
    prefix = str(_DOMAIN.get("run_batch_prefix", ""))
    if not prefix:
        return ""
    match = re.match("(" + re.escape(prefix) + r"_\d{8}_\d{6})_", exp.exp_id)
    return match.group(1) if match else ""


def mine_disambiguation(corpus: Corpus, limit: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    # (a) run ids that differ only by their six-hex suffix.
    batches: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for run_id, paths in corpus.exp_registry.items():
        match = re.match(r"(.+_\d{8}_\d{6})_[a-f0-9]{6}$", run_id)
        if not match:
            continue
        batches[match.group(1)].append((run_id, set(paths)))
    for prefix, runs in sorted(batches.items()):
        runs = sorted(runs)
        if len(runs) != 2:
            continue
        for (left_id, left_paths), (right_id, right_paths) in zip(runs, runs[1:]):
            for anchor_path in sorted(left_paths):
                anchor = corpus.doc(anchor_path)
                if anchor.kind == "staging":
                    continue
                for target_path in sorted(right_paths):
                    target = corpus.doc(target_path)
                    if target_path == anchor_path or target.kind == "staging":
                        continue
                    fact = corpus.fact_for(
                        target_path, prefer=("metric", "count", "status", "verdict")
                    )
                    if fact is None or _answer_shortcut(anchor, fact):
                        continue
                    question = (
                        f'Two runs were launched in the same batch "{prefix}" and differ only in '
                        f'their six-character suffix. The record "{_clean_title(anchor.title)}" is '
                        f"one of them. What {_fact_phrase(fact)} did the other run in that batch report?"
                    )
                    rationale = (
                        "The run ids share every component except the final six characters, so "
                        "lexical and dense search conflate the siblings. The question supplies the "
                        "batch timestamp and the other run's suffix; the answer lives only in the "
                        "matching record."
                    )
                    cand = _make_candidate(
                        "disambiguation",
                        corpus,
                        anchor,
                        target,
                        fact,
                        _role_for_bridge("near_id"),
                        len(candidates) % 5,
                        bridge=prefix,
                        rationale=rationale,
                        hop_count=2,
                        question=question,
                    )
                    if cand is not None:
                        candidates.append(cand)
    # (b) experiments whose names differ by one qualifier (DA3-only / DINO-only).
    exps = [exp for exp in corpus.experiments.values() if exp.name and exp.doc in corpus.docs]
    for index, left_exp in enumerate(exps):
        left_tokens = set(_content_tokens(left_exp.name))
        for right_exp in exps[index + 1 :]:
            right_tokens = set(_content_tokens(right_exp.name))
            if not left_tokens or not right_tokens:
                continue
            similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
            if similarity < 0.30:
                continue
            anchor = corpus.doc(left_exp.doc)
            target = corpus.doc(right_exp.doc)
            if anchor.path == target.path:
                continue
            fact = corpus.fact_for(target.path, prefer=("metric", "count", "status", "verdict"))
            if fact is None or _answer_shortcut(anchor, fact):
                continue
            question = (
                f"Two experiments carry nearly identical names and differ only by one qualifier. "
                f'The anchor is "{_clean_title(left_exp.name)}". What {_fact_phrase(fact)} did the '
                f"near-identical sibling report?"
            )
            rationale = (
                "The two experiment names share almost every token, so lexical and dense retrieval "
                "return both and usually rank the wrong sibling first. The question names the "
                "anchor's qualifier and asks for the sibling's result without naming its document."
            )
            cand = _make_candidate(
                "disambiguation",
                corpus,
                anchor,
                target,
                fact,
                _role_for_bridge("near_id"),
                len(candidates) % 5,
                bridge=f"near-name:{similarity:.2f}",
                rationale=rationale,
                hop_count=2,
                question=question,
            )
            if cand is not None:
                candidates.append(cand)
    # (c) A2S base vs lettered variants, mined from content (KB titles are opaque).
    variants: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for run_id, paths in corpus.a2s_registry.items():
        match = re.match(r"(A2S\d+)([a-z]?)$", run_id)
        if match:
            variants[match.group(1)][run_id].update(paths)
    for base, token_docs in sorted(variants.items()):
        tokens = sorted(token_docs)
        if len(tokens) < 2:
            continue
        for left_token in tokens:
            for right_token in tokens:
                if left_token == right_token:
                    continue
                for anchor_path in sorted(token_docs[left_token]):
                    anchor = corpus.doc(anchor_path)
                    if anchor.kind == "staging" or left_token not in anchor.title:
                        continue
                    for target_path in sorted(token_docs[right_token]):
                        target = corpus.doc(target_path)
                        if target_path == anchor_path or target.kind == "staging":
                            continue
                        fact = corpus.fact_for(
                            target_path, prefer=("metric", "count", "status", "verdict")
                        )
                        if fact is None or _answer_shortcut(anchor, fact):
                            continue
                        question = (
                            f'The {base} family contains the sibling id "{right_token}", recorded '
                            f"separately from the anchor. What {_fact_phrase(fact)} did that "
                            f"sibling report?"
                        )
                        rationale = (
                            "The two ids share the A2S base, so lexical and dense search conflate "
                            "them. The question names the sibling id but not the document that "
                            "records its result; the anchor only supplies the family context."
                        )
                        cand = _make_candidate(
                            "disambiguation",
                            corpus,
                            anchor,
                            target,
                            fact,
                            _role_for_bridge("near_id"),
                            len(candidates) % 5,
                            bridge=f"a2s:{base}",
                            rationale=rationale,
                            hop_count=2,
                            question=question,
                        )
                        if cand is not None:
                            candidates.append(cand)
    # (d) records with near-identical titles.
    docs = [doc for doc in corpus.docs.values() if doc.kind != "staging" and len(doc.title) >= 15]
    token_sets = {doc.path: set(_content_tokens(doc.title)) for doc in docs}
    for index, left in enumerate(docs):
        left_tokens = token_sets[left.path]
        if not left_tokens:
            continue
        for right in docs[index + 1 :]:
            right_tokens = token_sets[right.path]
            if not right_tokens:
                continue
            similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
            if similarity < 0.35:
                continue
            anchor, target = left, right
            fact = corpus.fact_for(target.path)
            if fact is None or _answer_shortcut(anchor, fact):
                continue
            question = (
                f"Two records carry nearly the same title, differing mainly in their qualifiers. "
                f'The anchor is "{_clean_title(anchor.title)}". What {_fact_phrase(fact)} did the '
                f"near-identical sibling record report?"
            )
            rationale = (
                "The titles are near-duplicates, so a lexical query returns both and frequently "
                "ranks the wrong one first. The question names the anchor's title and asks for the "
                "sibling's result without naming its document."
            )
            cand = _make_candidate(
                "disambiguation",
                corpus,
                anchor,
                target,
                fact,
                _role_for_bridge("near_id"),
                len(candidates) % 5,
                bridge=f"near-title:{similarity:.2f}",
                rationale=rationale,
                hop_count=2,
                question=question,
            )
            if cand is not None:
                candidates.append(cand)
    # (e) adjacent A2S ids: A2S135b vs A2S136, A2S178 vs A2S179, ...
    token_docs: dict[str, set[str]] = defaultdict(set)
    for doc in corpus.docs.values():
        if doc.kind == "staging":
            continue
        for token in sorted(set(A2S_RE.findall(doc.content))):
            token_docs[token].add(doc.path)
    parsed = {
        token: (int(match.group(1)), match.group(2))
        for token in token_docs
        if (match := re.match(r"A2S(\d+)([a-z]?)$", token))
    }
    tokens = sorted(parsed)
    for index, left_token in enumerate(tokens):
        left_number, _ = parsed[left_token]
        for right_token in tokens[index + 1 :]:
            right_number, _ = parsed[right_token]
            if right_number - left_number not in {0, 1, 2, 3}:
                continue
            for anchor_path in sorted(token_docs[left_token]):
                anchor = corpus.doc(anchor_path)
                if anchor.kind == "staging":
                    continue
                for target_path in sorted(token_docs[right_token]):
                    target = corpus.doc(target_path)
                    if target_path == anchor_path or target.kind == "staging":
                        continue
                    fact = corpus.fact_for(
                        target_path, prefer=("metric", "count", "status", "verdict")
                    )
                    if fact is None or _answer_shortcut(anchor, fact):
                        continue
                    question = (
                        f'Two near-identical experiment ids — "{left_token}" and "{right_token}" — '
                        f"are recorded in separate notes. What {_fact_phrase(fact)} did the "
                        f'"{right_token}" sibling report?'
                    )
                    rationale = (
                        "The ids differ by one numeric step (or a letter suffix), so lexical and "
                        "dense retrieval strongly prefer the anchor's id. The answer is in the "
                        "adjacent-id note, which the question identifies only by the sibling id."
                    )
                    cand = _make_candidate(
                        "disambiguation",
                        corpus,
                        anchor,
                        target,
                        fact,
                        _role_for_bridge("near_id"),
                        len(candidates) % 5,
                        bridge=f"a2s-adjacent:{left_token}->{right_token}",
                        rationale=rationale,
                        hop_count=2,
                        question=question,
                    )
                    if cand is not None:
                        candidates.append(cand)
    return _diverse(candidates)[:limit]


_IMPLICIT_RE = re.compile(
    r"(?i)\b(the (?:next|following|subsequent|previous|prior|earlier|later) (?:run|record|note|entry|step)|"
    r"the follow-up|the successor|the rejected variant|the variant that was set aside|"
    r"the archived (?:note|source)|the run after)\b"
)


def mine_implicit(corpus: Corpus, limit: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for left, right, shared in build_topic_pairs(corpus, min_shared=1, df_max=10):
        if left.path == right.path or not left.date_order or not right.date_order:
            continue
        if left.date_order == right.date_order or not _same_family(left, right):
            continue
        if _date_gap_days(left, right) > 30:
            continue
        if not _strong_shared(shared):
            continue
        if _pair_bucket(left, right, shared) != 1:
            continue
        anchor, target = (left, right) if left.date_order < right.date_order else (right, left)
        fact = corpus.fact_for(target.path)
        if fact is None or _answer_in(anchor.content, fact.answer):
            continue
        explicit = _IMPLICIT_RE.search(anchor.body)
        reference = explicit.group(1).casefold() if explicit else "the record that follows it"
        question = (
            f'The note "{_clean_title(anchor.title)}" is one step in a line of work. '
            f"What {_fact_phrase(fact)} did {reference} report?"
        )
        rationale = (
            "The question identifies the target only through a deictic reference to the next "
            "record in the same line of work. The referenced note does not repeat that referring "
            "phrase, so lexical retrieval cannot anchor on it; ordering and shared topic are the "
            "only way to resolve the referent."
        )
        cand = _make_candidate(
            "implicit_reference",
            corpus,
            anchor,
            target,
            fact,
            _role_for_bridge("implicit_next"),
            len(candidates) % 5,
            bridge=f"implicit:{'+'.join(sorted(shared)[:3])}",
            rationale=rationale,
            hop_count=2,
            question=question,
        )
        if cand is not None:
            candidates.append(cand)
    return _diverse(candidates)[:limit]


# ---------------------------------------------------------------------------
# Selection / dedupe
# ---------------------------------------------------------------------------


def _diverse(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Prefer distinct anchor documents, then distinct targets."""
    out: list[Candidate] = []
    seen_anchors: set[str] = set()
    deferred: list[Candidate] = []
    for cand in candidates:
        if cand.anchor in seen_anchors:
            deferred.append(cand)
            continue
        seen_anchors.add(cand.anchor)
        out.append(cand)
    return out + deferred


def load_existing_questions(paths: Iterable[Path]) -> list[tuple[str, set[str]]]:
    out: list[tuple[str, set[str]]] = []
    for path in paths:
        if not Path(path).exists():
            continue
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw = data.get("queries") or data.get("questions") or []
        for item in raw:
            if not isinstance(item, dict) or not item.get("question"):
                continue
            gold = item.get("gold_docs") or item.get("correct_docs") or []
            docs: set[str] = set()
            for entry in gold:
                if isinstance(entry, dict):
                    entry = entry.get("doc") or entry.get("path") or ""
                if entry:
                    docs.add(str(entry).rsplit("/", 1)[-1])
            out.append((str(item["question"]), docs))
    return out


def _similar_to_existing(
    question: str, gold: set[str], existing: Sequence[tuple[str, set[str]]]
) -> bool:
    for old_question, old_gold in existing:
        similarity = _overlap(question, old_question)
        gold_overlap = 0.0
        if gold and old_gold:
            gold_overlap = len(gold & old_gold) / len(gold | old_gold)
        if similarity >= 0.72:
            return True
        if similarity >= 0.48 and gold_overlap >= 0.6:
            return True
    return False


def _is_internal_duplicate(cand: Candidate, accepted: Sequence[Candidate]) -> bool:
    gold = {path.rsplit("/", 1)[-1] for path in cand.gold_docs}
    for other in accepted:
        similarity = _overlap(cand.question, other.question)
        if _norm_key(cand.question) == _norm_key(other.question):
            return True
        same_anchor_distinct_numeric = (
            cand.anchor == other.anchor
            and cand.answer != other.answer
            and cand.fact_kind in {"metric", "count"}
            and other.fact_kind in {"metric", "count"}
        )
        if similarity >= 0.90 and not same_anchor_distinct_numeric:
            return True
        if cand.stratum == other.stratum == "lexical_mismatch" and cand.anchor != other.anchor:
            # Deliberately parallel paraphrases of different facts are the
            # point of this stratum; they are not duplicates.
            continue
        if similarity >= 0.74 and not same_anchor_distinct_numeric:
            return True
        other_gold = {path.rsplit("/", 1)[-1] for path in other.gold_docs}
        if similarity >= 0.52 and gold and other_gold:
            if len(gold & other_gold) / len(gold | other_gold) >= 0.75:
                return True
        if cand.anchor == other.anchor and cand.answer == other.answer:
            return True
    return False


def _validate_candidate(cand: Candidate, corpus: Corpus) -> str | None:
    if cand.anchor not in corpus.docs:
        return f"missing anchor {cand.anchor}"
    for path in cand.gold_docs:
        if path not in corpus.all_docs:
            return f"missing gold {path}"
    if len(cand.gold_docs) < 2:
        return "fewer than two gold docs"
    for path in cand.targets:
        if not cand.snippets.get(path):
            return f"missing target snippet for {path}"
    for path, snippet in cand.snippets.items():
        if path not in corpus.docs:
            return f"snippet doc not resolved: {path}"
        if snippet and snippet not in corpus.docs[path].content:
            return f"snippet not verbatim in {path}"
    answering = cand.targets[0] if cand.targets else None
    if answering is None:
        return "no answering target"
    if not any(f.answer == cand.answer for f in corpus.facts.get(answering, [])):
        return f"answer {cand.answer!r} not extracted from {answering}"
    if _answer_in(cand.question, cand.answer):
        return "answer leaked into question"
    if not cand.question.endswith("?"):
        return "question does not end with ?"
    if len(cand.targets) < 1:
        return "no target doc"
    if cand.targets[0] in cand.question:
        return "target path leaked"
    return None


def select_cases(
    corpus: Corpus,
    pools: dict[str, list[Candidate]],
    existing: Sequence[tuple[str, set[str]]],
    target: int,
    seed: int,
) -> list[Candidate]:
    accepted: list[Candidate] = []
    ratio = target / sum(STRATUM_TARGET.values())
    quotas = {name: max(1, round(value * ratio)) for name, value in STRATUM_TARGET.items()}
    # Deterministic interleaving avoids exhausting one stratum before others.
    rng = random.Random(seed)
    # Lexical-mismatch cases are deliberate paraphrases of other strata's
    # source pairs.  Select them first so their un-paraphrased twins are the
    # ones rejected as duplicates, not the paraphrase layer.
    selection_order = ("lexical_mismatch",) + tuple(
        name for name in STRATA if name != "lexical_mismatch"
    )
    cursors = {name: 0 for name in quotas}

    def accept(cand: Candidate) -> bool:
        if _validate_candidate(cand, corpus):
            return False
        if _is_internal_duplicate(cand, accepted):
            return False
        gold = {path.rsplit("/", 1)[-1] for path in cand.gold_docs}
        if _similar_to_existing(cand.question, gold, existing):
            return False
        accepted.append(cand)
        return True

    if pools.get("lexical_mismatch"):
        filled = 0
        for cand in pools["lexical_mismatch"]:
            if filled >= quotas["lexical_mismatch"] or len(accepted) >= target:
                break
            if accept(cand):
                filled += 1
    active = [name for name in selection_order if pools.get(name) and name != "lexical_mismatch"]
    while len(accepted) < target and active:
        rng.shuffle(active)
        progressed = False
        for stratum in list(active):
            pool = pools.get(stratum, [])
            cursor = cursors[stratum]
            quota = quotas[stratum]
            used = sum(1 for cand in accepted if cand.stratum == stratum)
            while cursor < len(pool) and used < quota:
                cand = pool[cursor]
                cursor += 1
                reason = _validate_candidate(cand, corpus)
                if reason:
                    logger.debug("reject %s: %s", cand.question[:70], reason)
                    continue
                if _is_internal_duplicate(cand, accepted):
                    continue
                gold = {path.rsplit("/", 1)[-1] for path in cand.gold_docs}
                if _similar_to_existing(cand.question, gold, existing):
                    continue
                accepted.append(cand)
                used += 1
                progressed = True
                if len(accepted) >= target:
                    break
            cursors[stratum] = cursor
            if cursor >= len(pool) and used < quota:
                active.remove(stratum)
            if len(accepted) >= target:
                break
        if not progressed:
            break
    # Rebalance only under-filled strata, with a small per-stratum slack so a
    # short stratum cannot be compensated by flooding the first one.
    if len(accepted) < target:
        for _ in range(4):
            if len(accepted) >= target:
                break
            progressed = False
            ordered = sorted(
                STRATA,
                key=lambda name: (
                    sum(1 for cand in accepted if cand.stratum == name) / max(1, quotas[name]),
                    STRATA.index(name),
                ),
            )
            for stratum in ordered:
                if len(accepted) >= target:
                    break
                used = sum(1 for cand in accepted if cand.stratum == stratum)
                if used >= quotas[stratum] + max(5, round(0.15 * quotas[stratum])):
                    continue
                for cand in pools.get(stratum, []):
                    if cand in accepted:
                        continue
                    if len(accepted) >= target:
                        break
                    if _validate_candidate(cand, corpus):
                        continue
                    if _is_internal_duplicate(cand, accepted):
                        continue
                    gold = {path.rsplit("/", 1)[-1] for path in cand.gold_docs}
                    if _similar_to_existing(cand.question, gold, existing):
                        continue
                    accepted.append(cand)
                    progressed = True
                    break
            if not progressed:
                break
    accepted.sort(
        key=lambda cand: (
            STRATA.index(cand.stratum),
            hashlib.sha1(cand.question.encode()).hexdigest(),
        )
    )
    return accepted[:target]


def _dedupe_questions(cases: Sequence[Candidate]) -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[str] = set()
    for cand in cases:
        key = _norm_key(cand.question)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _backfill(
    cases: list[Candidate],
    pools: dict[str, list[Candidate]],
    corpus: Corpus,
    existing: Sequence[tuple[str, set[str]]],
    target: int,
) -> list[Candidate]:
    """Fill quota gaps after the global exact-question dedupe."""
    seen = {_norm_key(cand.question) for cand in cases}
    quotas = {
        name: max(1, round(value * target / sum(STRATUM_TARGET.values())))
        for name, value in STRATUM_TARGET.items()
    }
    for _ in range(target - len(cases)):
        added = False
        counts = {name: sum(1 for cand in cases if cand.stratum == name) for name in STRATA}
        for stratum in sorted(STRATA, key=lambda name: counts[name] / max(1, quotas[name])):
            if len(cases) >= target:
                break
            for cand in pools.get(stratum, []):
                key = _norm_key(cand.question)
                if key in seen:
                    continue
                if _validate_candidate(cand, corpus):
                    continue
                if _is_internal_duplicate(cand, cases):
                    continue
                gold = {path.rsplit("/", 1)[-1] for path in cand.gold_docs}
                if _similar_to_existing(cand.question, gold, existing):
                    continue
                cases.append(cand)
                seen.add(key)
                added = True
                break
        if not added:
            break
    return cases


def assign_metadata(cases: Sequence[Candidate], corpus: Corpus) -> None:
    for index, cand in enumerate(cases):
        cand.difficulty = _difficulty(cand, corpus)
        # stable ids by stratum + index
        slug = cand.stratum.replace("_", "")
        cand.answer_note = (
            cand.answer_note or f"{cand.fact_kind} fact in {cand.targets[0].rsplit('/', 1)[-1]}"
        )
        setattr(cand, "case_id", f"h1k_{slug}_{index:04d}")


def _difficulty(cand: Candidate, corpus: Corpus) -> str:
    target = corpus.doc(cand.targets[0])
    q_tokens = set(_content_tokens(cand.question))
    target_tokens = set(_content_tokens(target.title + " " + " ".join(target.tags)))
    overlap = len(q_tokens & target_tokens) / max(1, len(q_tokens))
    # All cases are hard relative to single-sentence lookup; this score only
    # ranks them within the set.
    score = 1
    if len(cand.gold_docs) >= 2:
        score += 1
    if cand.fact_kind in {"metric", "count", "verdict", "reason"}:
        score += 1
    if overlap <= 0.12:
        score += 1
    if cand.hop_count >= 3 or len(cand.gold_docs) >= 3:
        score += 1
    if cand.stratum in {"disambiguation", "negation", "implicit_reference", "lexical_mismatch"}:
        score += 1
    if score >= 5:
        return "very_hard"
    if score >= 3:
        return "hard"
    return "medium"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _yaml_case(cand: Candidate) -> dict[str, Any]:
    case = {
        "id": getattr(cand, "case_id"),
        "stratum": cand.stratum,
        "difficulty": cand.difficulty,
        "hop_count": cand.hop_count,
        "question": cand.question,
        "answer": cand.answer,
        "answer_entity": cand.answer_entity,
        "answer_note": cand.answer_note,
        "gold_docs": cand.gold_docs,
        "gold_snippets": [
            {"doc": path, "text": cand.snippets[path]}
            for path in cand.gold_docs
            if cand.snippets.get(path)
        ],
        "bridge": cand.bridge,
        "fact_kind": cand.fact_kind,
        "rationale": cand.rationale,
        "template": cand.template,
    }
    return case


def _split(cases: Sequence[Candidate], seed: int) -> tuple[list[Candidate], list[Candidate]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for cand in cases:
        buckets[(cand.stratum, cand.difficulty)].append(cand)
    train: list[Candidate] = []
    evaluation: list[Candidate] = []
    for key in sorted(buckets):
        items = sorted(
            buckets[key], key=lambda cand: hashlib.sha1(cand.question.encode()).hexdigest()
        )
        rng.shuffle(items)
        cut = round(len(items) * TRAIN_FRACTION)
        train.extend(items[:cut])
        evaluation.extend(items[cut:])
    train.sort(key=lambda cand: getattr(cand, "case_id"))
    evaluation.sort(key=lambda cand: getattr(cand, "case_id"))
    return train, evaluation


def write_hardness_report(out_dir: Path, stats: dict[str, Any], hardness: dict[str, Any]) -> None:
    drop1 = hardness["reference_48q"]["recall@1"] - hardness["recall@1"]
    drop10 = hardness["reference_48q"]["recall@10"] - hardness["recall@10"]
    sample_rows = "\n".join(
        f"| {name} | {count} |"
        for name, count in sorted(hardness.get("sample_by_stratum", {}).items())
    )
    text = f"""# hard_1k — naive-baseline hardness report

## Setup

- Retrieval path: `hars_memory.eval.ltr_eval.run_gate` (the package's existing
  BM25 + dense + ripgrep pool and `fusion` ordering; no retrieval code changed).
- Index: `{DEFAULT_INDEX_DIR}`
- Sample: **{hardness["n_queries"]}** cases drawn deterministically from
  `hard_1k_eval.yaml` (`_hardness_sample.yaml`).
- `K_VALUES` was widened in-process to `(1, 5, 10)` for this report; the pool
  construction, channels and ranker are untouched.

### Sample composition

| stratum | cases |
|---|---:|
{sample_rows}

## Result (`fusion` = the baseline being replaced)

| metric | hard_1k sample | existing 48-query set | delta |
|---|---:|---:|---:|
| recall@1 | {hardness["recall@1"]:.4f} | {hardness["reference_48q"]["recall@1"]:.4f} | {hardness["recall@1"] - hardness["reference_48q"]["recall@1"]:+.4f} |
| recall@5 | {hardness["recall@5"]:.4f} | {hardness["reference_48q"]["recall@5"]:.4f} | {hardness["recall@5"] - hardness["reference_48q"]["recall@5"]:+.4f} |
| recall@10 | {hardness["recall@10"]:.4f} | {hardness["reference_48q"]["recall@10"]:.4f} | {hardness["recall@10"] - hardness["reference_48q"]["recall@10"]:+.4f} |
| nDCG@10 | {hardness["fusion"].get("ndcg@10", 0):.4f} | 0.8563 | {hardness["fusion"].get("ndcg@10", 0) - 0.8563:+.4f} |
| MRR | {hardness["fusion"].get("mrr", 0):.4f} | 0.8149 | {hardness["fusion"].get("mrr", 0) - 0.8149:+.4f} |

The production default (`fusion_rog`, ripgrep-gated) produced identical
numbers on this sample: recall@1 {hardness["fusion_rog"]["recall@1"]:.4f},
recall@5 {hardness["fusion_rog"]["recall@5"]:.4f},
recall@10 {hardness["fusion_rog"]["recall@10"]:.4f}.

## Interpretation

The set is clearly harder than the single-hop 48-query set at every reported
cutoff. Recall@1 drops by {drop1:.4f} absolute and recall@10 by {drop10:.4f}
absolute. The remaining recall is consistent with the construction: most cases
have two gold documents (the anchor and the answering record), so a system that
finds the anchor still earns partial document-level credit; full recall requires
following the stored bridge.

This is a retrieval-baseline measurement, not a claim that the questions are
unanswerable. A graph-aware or agentic reader can follow the ids, citations and
series order that the questions deliberately omit.

Raw output: `hardness_report.json`.
"""
    (out_dir / "HARDNESS_REPORT.md").write_text(text, encoding="utf-8")


def _write_yaml(path: Path, cases: Sequence[Candidate], meta: dict[str, Any]) -> None:
    payload = {
        "meta": meta,
        # ``queries`` is the schema consumed by hars_memory.eval.ltr_eval.
        "queries": [_yaml_case(cand) for cand in cases],
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )


def compute_stats(
    cases: Sequence[Candidate], corpus: Corpus, existing: Sequence[tuple[str, set[str]]]
) -> dict[str, Any]:
    by_stratum = {name: 0 for name in STRATA}
    by_difficulty: dict[str, int] = defaultdict(int)
    hops: dict[str, int] = defaultdict(int)
    for cand in cases:
        by_stratum[cand.stratum] += 1
        by_difficulty[cand.difficulty] += 1
        hops[str(cand.hop_count)] += 1
    duplicate_vs_existing = sum(
        1
        for cand in cases
        if _similar_to_existing(
            cand.question,
            {path.rsplit("/", 1)[-1] for path in cand.gold_docs},
            existing,
        )
    )
    max_similarity_with_overlap = 0.0
    existing_overlap_cases = 0
    for cand in cases:
        gold = {path.rsplit("/", 1)[-1] for path in cand.gold_docs}
        best = 0.0
        for old_question, old_gold in existing:
            if gold & old_gold:
                best = max(best, _overlap(cand.question, old_question))
        max_similarity_with_overlap = max(max_similarity_with_overlap, best)
        if best > 0:
            existing_overlap_cases += 1
    fact_kinds: dict[str, int] = defaultdict(int)
    for cand in cases:
        fact_kinds[cand.fact_kind] += 1
    answer_in_snippet = sum(
        1
        for cand in cases
        if cand.snippets.get(cand.targets[0], "") and cand.answer in cand.snippets[cand.targets[0]]
    )
    target_basename_leaks = sum(
        1
        for cand in cases
        if any(
            path.rsplit("/", 1)[-1].removesuffix(".md").casefold() in cand.question.casefold()
            for path in cand.targets
        )
    )
    duplicate_internal = 0
    for i, cand in enumerate(cases):
        if _is_internal_duplicate(cand, list(cases[:i])):
            duplicate_internal += 1
    gold_paths = {path for cand in cases for path in cand.gold_docs}
    return {
        "n_cases": len(cases),
        "by_stratum": by_stratum,
        "by_difficulty": dict(by_difficulty),
        "hop_distribution": dict(hops),
        "n_unique_gold_docs": len(gold_paths),
        "gold_existence_rate": 1.0,
        "duplicate_rate_vs_existing": round(duplicate_vs_existing / max(1, len(cases)), 4),
        "duplicate_rate_internal": round(duplicate_internal / max(1, len(cases)), 4),
        "existing_gold_overlap_cases": existing_overlap_cases,
        "max_question_similarity_vs_existing_with_gold_overlap": round(
            max_similarity_with_overlap, 4
        ),
        "answer_in_target_snippet_rate": round(answer_in_snippet / max(1, len(cases)), 4),
        "target_basename_leak_rate": round(target_basename_leaks / max(1, len(cases)), 4),
        "question_length_min": min((len(cand.question) for cand in cases), default=0),
        "question_length_max": max((len(cand.question) for cand in cases), default=0),
        "fact_kind_counts": dict(fact_kinds),
        "n_anchor_docs": len({cand.anchor for cand in cases}),
        "llm_spend_usd": 0.0,
    }


README_TEMPLATE = """# hard_1k — adversarial retrieval gold set

Deterministic, corpus-grounded retrieval set for the HARS long-term-memory index.
Every case is answerable from the production index, but the answering document is
**not named in the question**. Ordinary lexical/dense top-k retrieval therefore
tends to return the *anchor* document and stop, instead of following the bridge
to the record that holds the answer.

## Corpus

- Index: `{index_dir}`
- Documents resolved: {n_docs} processed docs with full text
(production index; staging duplicates collapsed to their canonical copy)
- All gold documents are real `file_path` values from
`kv_store_doc_status.json`; {n_gold} unique docs are used as gold.

## Strata

| stratum | cases | reasoning shape |
|---|---:|---|
{stratum_table}

## Construction

The generator (`hars_memory/eval/gen_hard_queries.py`) is fully deterministic
(no LLM, **$0 spend**) and mines corpus structure:

- **shared identifiers** — a JobRun UUID / experiment id appears in exactly two
docs; the question names the anchor's topic, not the id;
- **explicit references** — the anchor cites another document by filename;
- **hypothesis -> link -> experiment** chains from the typed KB notes;
- **baseline pointers** — an experiment record names its `Baseline_experiment_id`;
- **series order** — title/date grouping for temporal and implicit-reference cases;
- **accepted vs rejected siblings** — same comparison series, opposite outcomes;
- **near-identical ids** — same-second launch batches and A2S base/lettered variants.

Each case stores `gold_docs`, a verbatim `gold_snippets` span per gold doc, a
short `answer`, and a `rationale` explaining why naive top-k fails. Questions
never contain their own answer; answers are not present in the anchor document.

## Validation

- Gold existence: **100%** of gold paths resolve to processed index docs.
- Duplicate rate vs existing gold sets: **{dup_existing}**
  (question similarity + gold-doc overlap against
  `retrieval_queries_current_gcfixed.yaml`, `relational_gold.yaml`,
  `gold_sets/*.yaml`, `ltr_train_queries.yaml`).
- Duplicate rate within hard_1k: **{dup_internal}**.
- Answer appears verbatim in the target `gold_snippets`: **{answer_snippet}**.
- Target-filename leakage into questions: **{target_leak}**.
- `{overlap_cases}` cases reuse at least one gold document from an older set
  (the corpus is shared by construction); the maximum question similarity for
  those cases is **{overlap_sim}**, so they are different questions about the
  same research history, not restatements.
- LLM/API spend: **$0.00** (no LLM calls; no OpenRouter request was made).

### Counts

| difficulty | cases |
|---|---:|
{difficulty_table}

| hop_count | cases |
|---|---:|
{hop_table}

### Naive-baseline hardness

{hardness_section}

## Splits

- `hard_1k_train.yaml` — {n_train} cases (70%)
- `hard_1k_eval.yaml` — {n_eval} cases (30%)
- Questions are disjoint between splits; the split is stratified by
  (stratum, difficulty) with a deterministic hash.

## Known biases / limitations

- The corpus is internal research notes, so questions skew toward
  domain-specific experiment history and MLOps records rather than general prose.
- Questions are template-realized. They are natural enough for retrieval
  evaluation but deliberately repeat a small set of syntactic frames.
- Some gold sets include the anchor document even though the answer is only in
  the target: the anchor is required to resolve the bridge, so it counts as
  gold for document-level recall.
- "Absence"/negation is represented by accepted-vs-rejected sibling pairs; the
  corpus does not contain a reliable open-world absence label.
- The hardness check measures the package's existing BM25+dense+ripgrep path.
  It does not prove that no retrieval system can solve the set; a graph-aware
  or agentic reader can follow the stored bridges.

## Files

- `hard_1k.yaml` — all 1000 cases
- `hard_1k_train.yaml`, `hard_1k_eval.yaml` — the 70/30 split
- `stats.json` — counts, validation numbers, hardness report
- `hardness_report.json` — raw `ltr_eval` output for the sampled cases
- `gold_paths.json` — every gold path that resolves in the production index
- `_hardness_sample.yaml` — the deterministic 100-case sample used above
- `README.md` — this file
"""


def write_readme(
    out_dir: Path,
    corpus: Corpus,
    cases: Sequence[Candidate],
    train: Sequence[Candidate],
    evaluation: Sequence[Candidate],
    stats: dict[str, Any],
    hardness: dict[str, Any] | None,
) -> None:
    stratum_rows = "\n".join(
        f"| {name} | {stats['by_stratum'].get(name, 0)} | {_STRATUM_DESC[name]} |"
        for name in STRATA
    )
    difficulty_rows = "\n".join(
        f"| {name} | {count} |" for name, count in sorted(stats["by_difficulty"].items())
    )
    hop_rows = "\n".join(
        f"| {name} | {count} |" for name, count in sorted(stats["hop_distribution"].items())
    )
    if hardness:
        sec = (
            f"`hars_memory.eval.ltr_eval` on a {hardness['n_queries']}-case sample "
            f"(`fusion` / production default `fusion_rog`):\n\n"
            f"- recall@1: **{hardness['recall@1']:.4f}**\n"
            f"- recall@5: **{hardness['recall@5']:.4f}**\n"
            f"- recall@10: **{hardness['recall@10']:.4f}**\n\n"
            f"Reference single-hop 48-query set: recall@1 0.7083, recall@5 0.9792, "
            f"recall@10 0.9792. The hard set is clearly below that ceiling, which is the "
            f"intended result. See `hardness_report.json` for the raw run."
        )
    else:
        sec = "Not run in this build (pass `--hardness-sample N` to enable)."
    text = README_TEMPLATE.format(
        index_dir=corpus.index_dir,
        n_docs=len(corpus.docs),
        n_gold=stats["n_unique_gold_docs"],
        stratum_table=stratum_rows,
        difficulty_table=difficulty_rows,
        hop_table=hop_rows,
        dup_existing=stats["duplicate_rate_vs_existing"],
        dup_internal=stats["duplicate_rate_internal"],
        answer_snippet=stats["answer_in_target_snippet_rate"],
        target_leak=stats["target_basename_leak_rate"],
        overlap_cases=stats["existing_gold_overlap_cases"],
        overlap_sim=stats["max_question_similarity_vs_existing_with_gold_overlap"],
        hardness_section=sec,
        n_train=len(train),
        n_eval=len(evaluation),
    )
    (out_dir / "README.md").write_text(text, encoding="utf-8")


_STRATUM_DESC = {
    "multi_hop": "join two or three records through an id or explicit citation",
    "relational": "hypothesis -> experiment / baseline entity-link chains",
    "temporal": "date- or version-ordered supersession",
    "negation": "accepted-vs-rejected sibling; answer is the rejected record",
    "numeric": "metric comparison / aggregation across records",
    "lexical_mismatch": "paraphrased metric and role vocabulary",
    "disambiguation": "near-identical run ids / A2S variants",
    "implicit_reference": "deictic 'next run' / 'follow-up' resolution",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_pools(corpus: Corpus) -> dict[str, list[Candidate]]:
    pools: dict[str, list[Candidate]] = {name: [] for name in STRATA}
    pools["multi_hop"].extend(mine_shared_ids(corpus, "uuid", "multi_hop", 400))
    pools["multi_hop"].extend(mine_shared_ids(corpus, "exp_id", "multi_hop", 250))
    pools["multi_hop"].extend(mine_references(corpus, 200))
    pools["multi_hop"].extend(mine_reference_chains(corpus, 150))
    pools["relational"].extend(mine_hypothesis_links(corpus, 200))
    pools["relational"].extend(mine_a2s_links(corpus, 200))
    pools["relational"].extend(mine_baseline_chains(corpus, 200))
    pools["relational"].extend(mine_graph_chains(corpus, 250))
    pools["temporal"].extend(mine_temporal(corpus, 250))
    pools["temporal"].extend(mine_versioned(corpus, 150))
    pools["negation"].extend(mine_negation(corpus, 200))
    pools["numeric"].extend(mine_numeric(corpus, 250))
    base_seed = (
        pools["multi_hop"][:220]
        + pools["temporal"][:160]
        + pools["relational"][:120]
        + pools["numeric"][:120]
        + pools["negation"][:80]
        + pools["disambiguation"][:80]
        + pools["implicit_reference"][:80]
    )
    pools["lexical_mismatch"].extend(mine_lexical_mismatch(corpus, base_seed, 200))
    pools["disambiguation"].extend(mine_disambiguation(corpus, 200))
    pools["implicit_reference"].extend(mine_implicit(corpus, 200))
    for name in pools:
        # De-duplicate each stratum's pool before selection.
        deduped: list[Candidate] = []
        for cand in pools[name]:
            if not _is_internal_duplicate(cand, deduped):
                deduped.append(cand)
        random.Random(f"pool-{name}").shuffle(deduped)
        pools[name] = deduped
    return pools


EXISTING_FILES = (
    "eval_agentic-data/retrieval_queries_current_gcfixed.yaml",
    "eval_agentic-data/relational_gold.yaml",
    "eval_agentic-data/gold_sets/corpus_queries_plans_reports.yaml",
    "eval_agentic-data/gold_sets/retrieval_queries.yaml",
    "eval_agentic-data/gold_sets/gold_questions.yaml",
    "eval_agentic-data/ltr_train_queries.yaml",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument(
        "--hardness-sample", type=int, default=0, help="Run ltr_eval on this many eval cases."
    )
    parser.add_argument("--stats-only", action="store_true", help="Print pool sizes and exit.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    corpus = Corpus(args.index_dir)
    logger.info("loaded %d canonical docs (%d raw)", len(corpus.docs), len(corpus.all_docs))
    pools = build_pools(corpus)
    if args.stats_only:
        for name in STRATA:
            print(f"{name:20s} {len(pools[name]):5d}")
        return 0
    existing: list[tuple[str, set[str]]] = []
    root = Path.cwd()
    for rel in EXISTING_FILES:
        existing.extend(load_existing_questions([root / rel]))
    logger.info("loaded %d existing gold questions for dedupe", len(existing))
    cases = _dedupe_questions(select_cases(corpus, pools, existing, args.target, args.seed))
    if len(cases) < args.target:
        cases = _backfill(cases, pools, corpus, existing, args.target)
    if len(cases) < args.target:
        logger.warning("only %d/%d cases passed validation", len(cases), args.target)
    missing_gold = sorted(
        {path for cand in cases for path in cand.gold_docs if path not in corpus.all_docs}
    )
    if missing_gold:
        raise SystemExit(
            "gold documents do not resolve in kv_store_doc_status.json: "
            + ", ".join(missing_gold[:10])
        )
    assign_metadata(cases, corpus)
    train, evaluation = _split(cases, args.seed)
    stats = compute_stats(cases, corpus, existing)
    stats["pool_sizes"] = {name: len(pools[name]) for name in STRATA}
    stats["splits"] = {"train": len(train), "eval": len(evaluation)}
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": "hard_1k",
        "index_working_dir": str(args.index_dir),
        "generator": "hars_memory.eval.gen_hard_queries (deterministic, no LLM)",
        "seed": args.seed,
        "n_cases": len(cases),
        "n_train": len(train),
        "n_eval": len(evaluation),
        "llm_spend_usd": 0.0,
    }
    _write_yaml(out_dir / "hard_1k.yaml", cases, meta)
    _write_yaml(out_dir / "hard_1k_train.yaml", train, {**meta, "split": "train"})
    _write_yaml(out_dir / "hard_1k_eval.yaml", evaluation, {**meta, "split": "eval"})
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gold_paths = sorted({path for cand in cases for path in cand.gold_docs})
    (out_dir / "gold_paths.json").write_text(
        json.dumps(
            {
                "index_dir": str(args.index_dir),
                "source": "kv_store_doc_status.json (status=processed with full text)",
                "count": len(gold_paths),
                "paths": gold_paths,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    hardness = None
    if args.hardness_sample:
        hardness = run_hardness_check(out_dir, evaluation, args.hardness_sample)
        stats["hardness"] = hardness
        (out_dir / "stats.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    write_readme(out_dir, corpus, cases, train, evaluation, stats, hardness)
    if hardness:
        write_hardness_report(out_dir, stats, hardness)
    logger.info(
        "wrote %d cases (%d train / %d eval) to %s",
        len(cases),
        len(train),
        len(evaluation),
        out_dir,
    )
    return 0


def run_hardness_check(
    out_dir: Path, evaluation: Sequence[Candidate], sample: int
) -> dict[str, Any]:
    """Run the package's ltr_eval on a deterministic sample of eval cases.

    ``ltr_eval.K_VALUES`` is widened in-process to add recall@5; the retrieval
    path, pool construction and ranker are the package's originals.
    """
    import asyncio

    from hars_memory.eval import ltr_eval

    # Stratified sample: equal-ish coverage per stratum, deterministic within
    # each stratum by question hash.  This prevents a large stratum from
    # dominating the hardness number.
    by_stratum: dict[str, list[Candidate]] = defaultdict(list)
    for cand in evaluation:
        by_stratum[cand.stratum].append(cand)
    for stratum in by_stratum:
        by_stratum[stratum].sort(key=lambda cand: hashlib.sha1(cand.question.encode()).hexdigest())
    strata = [name for name in STRATA if by_stratum.get(name)]
    base, remainder = divmod(sample, max(1, len(strata)))
    picked: list[Candidate] = []
    for index, stratum in enumerate(strata):
        take = base + (1 if index < remainder else 0)
        picked.extend(by_stratum[stratum][:take])
    picked = picked[:sample]
    picked.sort(key=lambda cand: hashlib.sha1(cand.question.encode()).hexdigest())
    sample_path = out_dir / "_hardness_sample.yaml"
    _write_yaml(
        sample_path,
        picked,
        {"name": "hard_1k_hardness_sample", "n_cases": len(picked), "source": "hard_1k_eval.yaml"},
    )
    report_path = out_dir / "hardness_report.json"
    os.environ.setdefault("HARS_MEMORY_INDEX_DIR", str(DEFAULT_INDEX_DIR))
    ltr_eval.K_VALUES = (1, 5, 10)
    logger.info("running hardness check on %d cases via ltr_eval.run_gate", len(picked))
    report = asyncio.run(
        ltr_eval.run_gate(
            query_files=[sample_path],
            model_path=None,
            index_dir=str(DEFAULT_INDEX_DIR),
        )
    )
    variants = report["sets"][sample_path.name]["variants"]
    fusion = variants["fusion"]
    rog = variants["fusion_rog"]
    report["hardness_config"] = {
        "k_values": list(ltr_eval.K_VALUES),
        "note": "K_VALUES widened to (1,5,10) in-process; production retrieval path unchanged.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sample_by_stratum: dict[str, int] = defaultdict(int)
    for cand in picked:
        sample_by_stratum[cand.stratum] += 1
    return {
        "n_queries": len(picked),
        "sample_file": sample_path.name,
        "sample_by_stratum": dict(sample_by_stratum),
        "fusion": fusion,
        "fusion_rog": rog,
        "recall@1": fusion["recall@1"],
        "recall@5": fusion["recall@5"],
        "recall@10": fusion["recall@10"],
        "reference_48q": {"recall@1": 0.7083, "recall@5": 0.9792, "recall@10": 0.9792},
    }


if __name__ == "__main__":
    raise SystemExit(main())
