#!/usr/bin/env python3
"""ONE token-counting layer for every parity-related number (Amendment C.5).

Why: parity and all budgets used to be measured with three different counters (tiktoken
cl100k in the runner/packer, chars/3.6 in the pi extension, chars/3.6 fallback) while the
generation caps (max_tokens) are NATIVE tokens of the answering model. On Claude models the
native count is ~1.5-1.65x cl100k (measured 2026-10-26: 1.53-1.64 on prose/code), so
"1536 of our tokens" of nc memory was ~1.5x more native tokens than the baseline summary got.

Rule: budgets, realized tokens, injected recall tokens and memory_tokens are measured in the
ANSWERING MODEL's native tokenizer:

  anthropic   POST {base}/v1/messages/count_tokens  {"model", "messages":[{"role":"user","content":text}]}
              -> {"input_tokens": N}.  Free (Anthropic docs: token counting is free, subject to
              requests-per-minute limits by usage tier).  The request frames the text as one user
              message, which adds a constant (8 tokens measured on claude-haiku-5-5 / sonnet-5-5);
              it is measured once per (provider, model) as count("x") - 1 and subtracted, so
              count(text) is the cost of the TEXT.
  gemini      POST {base}/v1beta/models/{model}:countTokens {"contents":[{"parts":[{"text":text}]}]}
              -> {"totalTokens": N}  (header x-goog-api-key; not billed).
  other       (OpenRouter, custom, no key) -> tiktoken cl100k (else chars/3.6), `native=False`.
  estimate    offline/dry-run/tests: fast estimator x a calibrated ratio, `native=False, estimated=True`.

Every count is cached on disk keyed by (provider, model, sha256(text)); batches run with a
concurrency limit; 429/5xx are retried with backoff.  `tokenizer_id` + `native` are stamped on
records and manifests.

    import tokcount
    ctr = tokcount.Counter("anthropic", "claude-haiku-5-5", mode="auto")   # key from ANTHROPIC_API_KEY
    ctr.count(text); ctr.tokenizer_id; ctr.native
    tokcount.set_default(ctr)          # run_chainreplay / run_longsession use the default counter
    k = ctr.max_prefix(text, 1536)     # largest cut that fits the native budget (binary search, cached)
    out, info = tokcount.fit_native(ctr, text, 1536, lambda t, est_cap: my_trimmer(t, est_cap))

CLI:  tokcount.py probe [--provider anthropic] [--model M] [--file F]   (doctor: reachability + ratio)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# --------------------------------------------------------------------- fast estimator
try:
    if os.environ.get("NC_TOKENIZER") == "estimator":       # tests / reproducible runs: never tiktoken
        raise ImportError("tiktoken disabled by NC_TOKENIZER=estimator")
    import tiktoken  # noqa: F401

    _ENC = tiktoken.get_encoding("cl100k_base")

    def fast_count(text: str) -> int:
        return len(_ENC.encode(text or ""))

    FAST_ID = "tiktoken:cl100k_base"
except Exception:  # tiktoken missing, or its vocabulary cannot be fetched
    def fast_count(text: str) -> int:
        return int(round(len(text or "") / 3.6))

    FAST_ID = "estimator:chars/3.6"

# native/fast ratio priors (used only before the first native sample, and by mode=estimate):
# measured 2026-10-26 on claude-haiku-5-5: native / (chars/3.6) = 1.58-1.60, native / cl100k = 1.42-1.64.
RATIO_PRIOR_BY_FAST = {
    "anthropic": {"tiktoken:cl100k_base": 1.55, "estimator:chars/3.6": 1.60},
    "gemini": {"tiktoken:cl100k_base": 1.00, "estimator:chars/3.6": 1.00},
}
NATIVE_PROVIDERS = ("anthropic", "gemini")
ANTHROPIC_VERSION = "2023-06-01"
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
CACHE_ENV = "NC_TOKCOUNT_CACHE"
MODE_ENV = "NC_TOKCOUNT"           # auto | native | tiktoken | estimate
RATIO_ENV = "NC_TOKCOUNT_RATIO"    # estimate-mode ratio override
_OVERHEAD_H = "overhead:x"


class TokCountError(RuntimeError):
    pass


def prior_ratio(provider: str) -> float:
    """Measured native/fast ratio prior for a provider (1.0 when unknown)."""
    return RATIO_PRIOR_BY_FAST.get((provider or "").lower(), {}).get(FAST_ID, 1.0)


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", "surrogatepass")).hexdigest()


def default_cache_path() -> str:
    return os.environ.get(CACHE_ENV) or os.path.join(os.path.expanduser("~"), ".cache", "nc-tokcount", "cache.jsonl")


class Counter:
    """Token counter for ONE (provider, model). Thread-safe."""

    def __init__(self, provider: str, model: str, *, mode: str | None = None, key: str | None = None,
                 key_env: str | None = None, base_url: str | None = None, cache_path: str | None = None,
                 transport=None, concurrency: int = 4, timeout: float = 60.0, ratio: float | None = None,
                 sleep=time.sleep, max_tries: int = 6):
        self.provider = (provider or "").lower()
        self.model = model or ""
        self.concurrency = max(1, int(concurrency))
        self.timeout = timeout
        self._sleep = sleep
        self.max_tries = max_tries
        self._transport = transport          # callable(text) -> raw count (tests / mock native); no network
        self._key = key
        self._key_env = key_env or KEY_ENV.get(self.provider)
        self._base_url = base_url
        self._lock = threading.RLock()
        self._sem = threading.BoundedSemaphore(self.concurrency)
        self._mem: dict[str, int] = {}
        self._cache_path = cache_path if cache_path is not None else default_cache_path()
        self._cache_f = None
        self.stats = {"calls": 0, "cache_hits": 0, "retries": 0, "api_seconds": 0.0, "texts": 0}
        self._sn = 0
        self._sf = 0
        self._overhead: int | None = None
        mode = (mode or os.environ.get(MODE_ENV) or "auto").lower()
        if mode not in ("auto", "native", "tiktoken", "estimate"):
            raise TokCountError(f"unknown tokcount mode {mode!r} (auto|native|tiktoken|estimate)")
        self.requested_mode = mode
        has_key = bool(self._key or (self._key_env and os.environ.get(self._key_env, "").strip()))
        can_native = self.provider in NATIVE_PROVIDERS and (self._transport is not None or has_key)
        if mode == "native":
            if self.provider not in NATIVE_PROVIDERS and self._transport is None:
                raise TokCountError(f"no native token counter for provider {provider!r}")
            if not can_native:
                raise TokCountError(f"native counting needs {self._key_env} (provider {provider})")
            self.mode = "native"
        elif mode == "auto":
            self.mode = "native" if can_native else "tiktoken"
        else:
            self.mode = mode
        prior = RATIO_PRIOR_BY_FAST.get(self.provider, {}).get(FAST_ID, 1.0)
        env_ratio = os.environ.get(RATIO_ENV)
        # estimate mode: ratio is EXPLICIT (arg / NC_TOKCOUNT_RATIO / prior_ratio()); default 1.0 = the fast
        # estimator unchanged (hermetic tests). Dry-run projections pass prior_ratio(provider).
        self.ratio = float(ratio if ratio is not None else (env_ratio or 1.0))
        self._prior = prior
        if self.mode == "native":
            self._load_cache()

    # ------------------------------------------------------------------ identity
    @property
    def native(self) -> bool:
        return self.mode == "native"

    @property
    def estimated(self) -> bool:
        return self.mode == "estimate"

    @property
    def tokenizer_id(self) -> str:
        if self.mode == "native":
            api = {"anthropic": "count_tokens", "gemini": "countTokens"}.get(self.provider, "native")
            return f"{self.provider}:{api}:{self.model}"
        if self.mode == "estimate":
            return f"estimator:{FAST_ID}*{self.ratio:g}"
        return FAST_ID

    @property
    def scale(self) -> float:
        """Observed native/fast ratio (running, over every text counted so far); the prior before samples."""
        if self.mode == "estimate":
            return self.ratio
        if self.mode == "native" and self._sf >= 400:
            return self._sn / self._sf
        return self._prior if self.mode == "native" else 1.0

    def manifest(self) -> dict:
        return {"tokenizer": self.tokenizer_id, "native": self.native, "estimated": self.estimated,
                "provider": self.provider, "model": self.model, "mode": self.mode,
                "fast_estimator": FAST_ID, "scale_native_over_fast": round(self.scale, 4),
                "overhead_subtracted": self._overhead, "calls": self.stats["calls"],
                "cache_hits": self.stats["cache_hits"], "retries": self.stats["retries"],
                "cache_path": self._cache_path if self.native else None}

    # --------------------------------------------------------------------- cache
    def _ck(self, h: str) -> str:
        return f"{self.provider}|{self.model}|{h}"

    def _load_cache(self) -> None:
        p = self._cache_path
        if not p or not os.path.isfile(p):
            return
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if d.get("p") == self.provider and d.get("m") == self.model and "h" in d and "n" in d:
                        self._mem[self._ck(d["h"])] = int(d["n"])
        except OSError:
            pass

    def _store(self, h: str, n: int) -> None:
        with self._lock:
            self._mem[self._ck(h)] = n
            p = self._cache_path
            if not p:
                return
            try:
                if self._cache_f is None:
                    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
                    self._cache_f = open(p, "a", encoding="utf-8")
                self._cache_f.write(json.dumps({"p": self.provider, "m": self.model, "h": h, "n": n}) + "\n")
                self._cache_f.flush()
            except OSError:
                self._cache_f = None      # cache is an optimisation, never a failure

    def close(self) -> None:
        with self._lock:
            if self._cache_f:
                self._cache_f.close()
                self._cache_f = None

    # ----------------------------------------------------------------------- API
    def _url_headers_body(self, text: str):
        if self.provider == "anthropic":
            base = (self._base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            body = {"model": self.model, "messages": [{"role": "user", "content": text}]}
            hdr = {"x-api-key": self._api_key(), "anthropic-version": ANTHROPIC_VERSION,
                   "content-type": "application/json"}
            return base + "/v1/messages/count_tokens", hdr, body
        base = (self._base_url or os.environ.get("GEMINI_BASE_URL")
                or "https://generativelanguage.googleapis.com").rstrip("/")
        model = self.model if self.model.startswith("models/") else f"models/{self.model}"
        body = {"contents": [{"parts": [{"text": text}]}]}
        hdr = {"x-goog-api-key": self._api_key(), "content-type": "application/json"}
        return f"{base}/v1beta/{model}:countTokens", hdr, body

    def _api_key(self) -> str:
        k = self._key or os.environ.get(self._key_env or "", "").strip()
        if not k:
            raise TokCountError(f"{self._key_env} is not set")
        return k

    def _raw(self, text: str) -> int:
        """One uncached native count (includes the request's framing overhead)."""
        if self._transport is not None:
            with self._sem:
                self.stats["calls"] += 1
                return int(self._transport(text))
        url, hdr, body = self._url_headers_body(text)
        data = json.dumps(body).encode("utf-8")
        last = ""
        for attempt in range(self.max_tries):
            with self._sem:
                t0 = time.time()
                try:
                    req = urllib.request.Request(url, data=data, headers=hdr, method="POST")
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        j = json.loads(r.read().decode("utf-8"))
                    self.stats["calls"] += 1
                    self.stats["api_seconds"] += time.time() - t0
                    n = j.get("input_tokens") if self.provider == "anthropic" else j.get("totalTokens")
                    if not isinstance(n, int):
                        raise TokCountError(f"unexpected count response keys {sorted(j)}")
                    return n
                except urllib.error.HTTPError as e:
                    code = e.code
                    detail = ""
                    try:
                        detail = e.read().decode("utf-8", "replace")[:200]
                    except Exception:  # noqa: BLE001
                        pass
                    last = f"HTTP {code} {detail}"
                    retry_after = e.headers.get("retry-after") if e.headers else None
                    if code not in (408, 409, 425, 429, 500, 502, 503, 504, 529):
                        raise TokCountError(f"count_tokens {self.provider}/{self.model}: {last}") from None
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                    last = f"{type(e).__name__}: {str(e)[:120]}"
                    retry_after = None
            self.stats["retries"] += 1
            try:
                delay = min(30.0, float(retry_after)) if retry_after else min(30.0, 1.5 * 2 ** attempt)
            except (TypeError, ValueError):
                delay = min(30.0, 1.5 * 2 ** attempt)
            self._sleep(delay)
        raise TokCountError(f"count_tokens {self.provider}/{self.model} failed after {self.max_tries} tries: {last}")

    def _ovh(self) -> int:
        """Constant framing overhead of the request (Anthropic: user-message wrapper), measured once."""
        if self.provider != "anthropic":
            return 0
        with self._lock:
            if self._overhead is None:
                h = _OVERHEAD_H
                raw = self._mem.get(self._ck(h))
                if raw is None:
                    raw = self._raw("x")
                    self._store(h, raw)
                self._overhead = max(0, raw - 1)         # "x" is one token
            return self._overhead

    # --------------------------------------------------------------------- public
    def count(self, text: str) -> int:
        text = text or ""
        if not text:
            return 0
        self.stats["texts"] += 1
        if self.mode == "tiktoken":
            return fast_count(text)
        if self.mode == "estimate":
            return max(1, int(round(fast_count(text) * self.ratio)))
        if not text.strip():                          # the API rejects whitespace-only content
            return max(1, int(round(fast_count(text) * self.scale)))
        h = sha256_text(text)
        k = self._ck(h)
        with self._lock:
            hit = self._mem.get(k)
        if hit is not None:
            self.stats["cache_hits"] += 1
            n = hit
        else:
            ovh = self._ovh()
            n = max(1, self._raw(text) - ovh)
            self._store(h, n)
        with self._lock:
            self._sn += n
            self._sf += fast_count(text)
        return n

    def count_many(self, texts) -> list[int]:
        """Counts for many texts; unique uncached ones go out with at most `concurrency` in flight."""
        texts = [t or "" for t in texts]
        if self.mode != "native":
            return [self.count(t) for t in texts]
        uniq = list(dict.fromkeys(t for t in texts if t))
        todo = [t for t in uniq if self._ck(sha256_text(t)) not in self._mem and t.strip()]
        if todo and self.provider == "anthropic":
            self._ovh()                               # measure the overhead once, before the pool
        if len(todo) > 1:
            with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
                list(ex.map(self.count, todo))
        return [self.count(t) for t in texts]

    def fast(self, text: str) -> int:
        return fast_count(text)

    def scale_for(self, text: str, sample_chars: int = 4000) -> float:
        """native/fast ratio measured on a prefix of `text` (one cached call); falls back to the running scale."""
        if self.mode != "native":
            return self.scale
        s = (text or "")[:sample_chars]
        f = fast_count(s)
        if f < 40:
            return self.scale
        return max(0.2, self.count(s) / f)

    def max_prefix(self, text: str, budget: int, suffix: str = "") -> int:
        """Largest k such that count(text[:k] + suffix) <= budget (binary search over cached counts)."""
        if budget <= 0:
            return 0
        if self.count(text + suffix) <= budget:
            return len(text)
        lo, hi = 0, len(text)
        # a Claude token is >= ~0.5 chars on code/ids; bound the search window from the observed scale
        guess = int(len(text) * budget / max(1, self.count(text + suffix)))
        lo_g, hi_g = max(0, int(guess * 0.8) - 16), min(len(text), int(guess * 1.25) + 16)
        if self.count(text[:lo_g] + suffix) <= budget:
            lo = lo_g
            if self.count(text[:hi_g] + suffix) > budget:
                hi = hi_g
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.count(text[:mid] + suffix) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def trim_to_budget(self, text: str, budget: int, marker: str = "") -> str:
        """text cut (at a line, else word boundary) so that cut + marker fits `budget` native tokens."""
        if self.count(text) <= budget:
            return text
        k = self.max_prefix(text, budget, marker)
        cut = text[:k]
        nl = cut.rfind("\n")
        if nl > 0:
            cut = cut[:nl]
        else:
            sp = cut.rfind(" ")
            if sp > 0:
                cut = cut[:sp]
        cut = cut.rstrip()
        while cut and self.count(cut + marker) > budget:          # snapping only shortens; belt and braces
            cut = cut[: int(len(cut) * 0.98)].rstrip()
        return cut + marker


# --------------------------------------------------------------------- fit helper
def fit_native(counter: "Counter | None", text: str, budget: int, trim_fn, max_iter: int = 14):
    """Hard-trim to a NATIVE budget with a trimmer that works in fast-estimator units.

    trim_fn(text, est_cap) -> (trimmed_text, info) must trim so fast_count(result) <= est_cap.
    The estimator cap starts at budget / (native/fast ratio of THIS text) and is lowered until
    the native count of the result fits (verify, don't assume). Without a counter (or a
    non-native one) the trimmer simply runs at `budget`. info gains native_tokens / native_over_cap /
    tokenizer / trim_iterations.
    """
    if counter is None or counter.mode == "tiktoken":
        out, info = trim_fn(text, budget)
        return out, info
    ratio = counter.scale_for(text) if counter.native else counter.scale
    est_cap = max(1, int(budget / max(ratio, 0.05)))
    out, info = trim_fn(text, est_cap)
    n = counter.count(out)
    it = 0
    while n > budget and it < max_iter and est_cap > 1:
        it += 1
        est_cap = max(1, min(est_cap - 1, int(est_cap * budget / n * 0.985)))
        out, info = trim_fn(text, est_cap)
        n = counter.count(out)
    info = dict(info or {})
    info.update({"native_tokens": n, "native_over_cap": n > budget, "tokenizer": counter.tokenizer_id,
                 "trim_iterations": it, "est_cap": est_cap})
    if "after" in info:
        info["after_fast"] = info["after"]
        info["after"] = n
    if "over_cap" in info or n > budget:
        info["over_cap"] = n > budget
    return out, info


# ---------------------------------------------------------------------- default
_DEFAULT: Counter | None = None
_DEFAULT_LOCK = threading.Lock()


def set_default(counter: "Counter | None") -> None:
    global _DEFAULT
    with _DEFAULT_LOCK:
        _DEFAULT = counter


def get_default() -> "Counter | None":
    return _DEFAULT


def count(text: str) -> int:
    """Budget-grade count with the default counter (fast estimator when none is configured)."""
    c = _DEFAULT
    return c.count(text) if c is not None else fast_count(text)


def provider_of_profile(profile_name: str | None, model: str | None = None) -> str:
    """tokcount provider for a profile name / model id: anthropic | gemini | other."""
    n = (profile_name or "").lower()
    if n in NATIVE_PROVIDERS:
        return n
    # OpenRouter / custom endpoints: the router's tokenizer is unknown, so never "native"
    return "other"


# ------------------------------------------------------------------------- CLI
def _probe(args) -> int:
    ctr = Counter(args.provider, args.model, mode=args.mode or "native", cache_path=args.cache,
                  key_env=args.key_env)
    texts = []
    if args.file:
        t = open(args.file, encoding="utf-8", errors="replace").read()
        texts.append((os.path.basename(args.file), t[: args.chars]))
    texts += [("hello", "hello world"), ("code", "def f(x):\n    return {'k': [x, x+1], 'id': 'a1b2c3'}\n" * 20)]
    rc = 0
    for name, t in texts:
        try:
            n = ctr.count(t)
        except TokCountError as e:
            print(f"FAIL {name}: {e}")
            rc = 1
            continue
        print(f"{name:12s} chars={len(t):6d} native={n:6d} fast({FAST_ID})={fast_count(t):6d} "
              f"native/fast={n / max(1, fast_count(t)):.3f} chars/native={len(t) / max(1, n):.2f}")
    print(json.dumps(ctr.manifest()))
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="count a few texts natively; prints the native/fast ratio (doctor check)")
    p.add_argument("--provider", default="anthropic")
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--mode", default=None)
    p.add_argument("--key-env", default=None, help="env var holding the key (default: the provider's own)")
    p.add_argument("--file", default=None)
    p.add_argument("--chars", type=int, default=6000)
    p.add_argument("--cache", default=None, help="cache file ('' = default location)")
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        return _probe(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
