"""Shared utilities for extraction SFT: chat rendering, tokenization,
output parsing, and parity metrics.

Import-safe without torch (stdlib + optional transformers) so the repo's
`uv run pytest` can test parsing/metrics without a GPU stack.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# LightRAG extraction record grammar (as produced by the 27B teacher):
#   entity<|#|>name<|#|>type<|#|>description
#   relation<|#|>source<|#|>target<|#|>keywords<|#|>description
#   <|COMPLETE|>       (terminal marker)
DELIM = "<|#|>"
COMPLETE = "<|COMPLETE|>"
ENTITY_FIELDS = 4  # entity | name | type | desc
RELATION_FIELDS = 5  # relation | src | tgt | keywords | desc
DEFAULT_ENTITY_TYPES = ["Concept", "Document", "Decision", "Event", "Person", "Tool", "Task"]


# --------------------------------------------------------------------------- #
# Chat rendering
# --------------------------------------------------------------------------- #

def render_prompt(tokenizer, user_content: str) -> str:
    """Render the user turn with a generation prompt (assistant header)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def render_full(tokenizer, user_content: str, assistant_content: str) -> str:
    """Render user+assistant turns exactly as the model should emit them."""
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        tokenize=False,
    )


def encode_completion_only(tokenizer, user_content: str, completion: str):
    """Token ids + labels where loss is applied only on the completion
    (assistant content + <|im_end|>), never on the prompt.

    The token list is *concatenated* (encode(prompt) + encode(completion) +
    [im_end]), guaranteeing an exact prefix property and stable masking.
    """
    prompt_text = render_prompt(tokenizer, user_content)
    ids_prompt = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    ids_out = tokenizer(completion, add_special_tokens=False)["input_ids"]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ids = ids_prompt + ids_out + [im_end]
    labels = [-100] * len(ids_prompt) + ids_out + [im_end]
    return ids, labels


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #

@dataclass
class ParsedOutput:
    entities: list = field(default_factory=list)   # (name, type, desc)
    relations: list = field(default_factory=list)  # (src, tgt, keywords, desc)
    complete: bool = False
    format_errors: list = field(default_factory=list)
    n_lines_total: int = 0


def parse_extraction_output(text: str) -> ParsedOutput:
    """Parse a LightRAG-style extraction output. Tolerates the teacher's
    rare noise lines (stray markup), which count as format errors only when
    they carry a record-looking prefix."""
    parsed = ParsedOutput()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parsed.n_lines_total += 1
        if line == COMPLETE or line.rstrip() == COMPLETE:
            parsed.complete = True
            continue
        if line.startswith(f"entity{DELIM}"):
            parts = line.split(DELIM)
            if len(parts) != ENTITY_FIELDS or any(not p.strip() for p in parts[1:]):
                parsed.format_errors.append(line[:160])
            else:
                parsed.entities.append(tuple(p.strip() for p in parts[1:]))
        elif line.startswith(f"relation{DELIM}"):
            parts = line.split(DELIM)
            if len(parts) != RELATION_FIELDS or any(not p.strip() for p in parts[1:4]):
                parsed.format_errors.append(line[:160])
            else:
                parsed.relations.append(tuple(p.strip() for p in parts[1:]))
        elif DELIM in line:
            # record-looking garbage: delimiter line with unknown prefix
            parsed.format_errors.append(line[:160])
        # anything else (prose/markup noise) is ignored, matching LightRAG's
        # lenient line parser
    return parsed


def extract_source_text(user_content: str) -> str:
    """Best-effort extraction of the `<Input Text>` block from the prompt."""
    marker = "<Input Text>"
    if marker in user_content:
        tail = user_content.split(marker, 1)[1]
        # strip fenced code block markers
        tail = tail.replace("```", "")
        return tail
    return user_content


def normalized(s: str) -> str:
    return " ".join(s.lower().split())


# --------------------------------------------------------------------------- #
# Parity metrics
# --------------------------------------------------------------------------- #

@dataclass
class ParityMetrics:
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0
    relation_precision: float = 0.0
    relation_recall: float = 0.0
    relation_f1: float = 0.0
    hallucination_rate: float = 0.0
    format_error_rate: float = 0.0   # fraction of samples with >=1 format error
    complete_rate: float = 0.0
    n_samples: int = 0


def _prf1(tp: int, pred: int, gold: int) -> tuple:
    p = tp / pred if pred else 0.0
    r = tp / gold if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def entity_f1_metrics(gold_entities, pred_entities):
    """F1 over (normalized name, normalized type) pairs, multiset semantics."""
    from collections import Counter

    g = Counter((normalized(n), normalized(t)) for n, t, *_ in gold_entities)
    p = Counter((normalized(n), normalized(t)) for n, t, *_ in pred_entities)
    tp = sum((g & p).values())
    return _prf1(tp, sum(p.values()), sum(g.values()))


def relation_f1_metrics(gold_relations, pred_relations):
    """F1 over (normalized source, normalized target) pairs (directional),
    multiset semantics. Keywords/description excluded (fuzzy by nature)."""
    from collections import Counter

    g = Counter((normalized(s), normalized(t)) for s, t, *_ in gold_relations)
    p = Counter((normalized(s), normalized(t)) for s, t, *_ in pred_relations)
    tp = sum((g & p).values())
    return _prf1(tp, sum(p.values()), sum(g.values()))


def hallucination_rate(pred_entities, source_text: str) -> float:
    """Fraction of predicted entity names whose normalized form does not
    appear (case-insensitive substring) in the source text."""
    if not pred_entities:
        return 0.0
    src = normalized(source_text)
    miss = 0
    for name, *_ in pred_entities:
        if normalized(name) not in src:
            miss += 1
    return miss / len(pred_entities)


def evaluate_pair(gold_text: str, pred_text: str, user_content: str = "") -> dict:
    gold = parse_extraction_output(gold_text)
    pred = parse_extraction_output(pred_text)
    ep, er, ef = entity_f1_metrics(gold.entities, pred.entities)
    rp, rr, rf = relation_f1_metrics(gold.relations, pred.relations)
    src = extract_source_text(user_content) if user_content else ""
    return {
        "entity_precision": ep, "entity_recall": er, "entity_f1": ef,
        "relation_precision": rp, "relation_recall": rr, "relation_f1": rf,
        "hallucination_rate": hallucination_rate(pred.entities, src),
        "n_format_errors": len(pred.format_errors),
        "complete": pred.complete,
        "gold": gold, "pred": pred,
    }


def aggregate(pairs: list) -> ParityMetrics:
    n = len(pairs)
    if not n:
        return ParityMetrics(n_samples=0)
    m = ParityMetrics(n_samples=n)
    m.entity_precision = sum(p["entity_precision"] for p in pairs) / n
    m.entity_recall = sum(p["entity_recall"] for p in pairs) / n
    m.entity_f1 = sum(p["entity_f1"] for p in pairs) / n
    m.relation_precision = sum(p["relation_precision"] for p in pairs) / n
    m.relation_recall = sum(p["relation_recall"] for p in pairs) / n
    m.relation_f1 = sum(p["relation_f1"] for p in pairs) / n
    m.hallucination_rate = sum(p["hallucination_rate"] for p in pairs) / n
    m.format_error_rate = sum(1 for p in pairs if p["n_format_errors"]) / n
    m.complete_rate = sum(1 for p in pairs if p["complete"]) / n
    return m


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(rows: list, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
