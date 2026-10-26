#!/usr/bin/env python3
"""Single provider layer for the neurocompact-eval scripts (vendor-agnostic).

A PROFILE (profiles/<name>.json: anthropic | openrouter | gemini | custom) defines, in
one place: base URL, auth header style, key env var, model id per ROLE, sampling-param
policy, pricing table, the `pi` provider/model style for the agent, and how the HARS
server's extractor/keyword LLM is configured. Switching vendor = `--profile NAME`
(or `--provider NAME`, the legacy alias); no code change.

Roles: agent, answer, summarizer, qa_generator, judge, extractor, keyword.
Per-role override on every script: `--model-<role> ID`. `--key-env VAR` reads the key
from VAR instead of the profile's variable (the value is never printed or logged).

Library API (stdlib only, offline-testable via an injectable transport):

    import providers as P
    prof = P.load_profile("anthropic")
    models = P.resolve_models(prof, {"judge": "claude-sonnet-5-5"})
    budget = P.Budget(2.50)                                # hard USD cap
    r = P.chat("judge", [{"role": "user", "content": "..."}], profile=prof,
               models=models, max_tokens=700, budget=budget)
    r.text, r.cost_usd, r.usage                            # ChatResult

CLI (no network, no keys):
    providers.py show      --profile anthropic             # role -> model, prices, pi flags
    providers.py hars-env  --profile anthropic             # HARS_MEMORY_* exports
    providers.py prices    --profile gemini
    providers.py check-family --profile anthropic          # agent vs judge verdict

Sampling policy: Anthropic 5.x rejects temperature/top_p -> `sampling.policy = "omit"`
(or `omit_for_families: ["anthropic"]` on a multi-vendor router); chat() drops them and
`hars-env` exports HARS_MEMORY_LLM_OMIT_SAMPLING_PARAMS=1.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.environ.get("NC_PROFILES_DIR") or os.path.join(os.path.dirname(BASE), "profiles")
ROLES = ("agent", "answer", "summarizer", "qa_generator", "judge", "extractor", "keyword")
SAMPLING_PARAMS = ("temperature", "top_p")
PROFILE_NAMES = ("anthropic", "openrouter", "gemini", "custom")


class ProviderError(RuntimeError):
    pass


class BudgetExceeded(ProviderError):
    pass


class UnpricedModel(ProviderError):
    pass


# --------------------------------------------------------------------- profiles

@dataclasses.dataclass
class Profile:
    name: str
    data: dict

    # ---- plain accessors
    @property
    def base_url(self) -> str | None:
        return self.data.get("base_url")

    @property
    def key_env(self) -> str:
        return self.data["key_env"]

    @property
    def pi(self) -> dict:
        return self.data.get("pi") or {}

    @property
    def family(self) -> str | None:
        return self.data.get("family")

    # ---- prompt caching (cost-only; Amendment C.3)
    @property
    def cache_cfg(self) -> dict:
        return self.data.get("cache") or {}

    @property
    def cache_enabled(self) -> bool:
        """Provider prompt caching ON for this profile (profile `cache.enabled`, --prompt-cache)."""
        return bool(self.cache_cfg.get("enabled")) and self.cache_cfg.get("api") == "anthropic_messages"

    def cache_control(self) -> dict:
        c = {"type": "ephemeral"}
        if str(self.cache_cfg.get("ttl") or "5m") == "1h":
            c["ttl"] = "1h"
        return c

    def default_model(self, role: str) -> str | None:
        return (self.data.get("models") or {}).get(role)

    # ---- sampling policy
    def omits_sampling(self, model: str | None) -> bool:
        s = self.data.get("sampling") or {}
        if s.get("policy") == "omit":
            return True
        fams = s.get("omit_for_families") or []
        return bool(model and fams and model_family(model) in fams)

    def sampling_kwargs(self, model: str | None, **params) -> dict:
        """Sampling params that may go on the wire for `model` (None values dropped)."""
        out = {k: v for k, v in params.items() if v is not None}
        if self.omits_sampling(model):
            for k in SAMPLING_PARAMS:
                out.pop(k, None)
        return out

    # ---- prices
    def price(self, model: str) -> dict | None:
        return (self.data.get("prices") or {}).get(model)

    def cost_usd(self, model: str, tokens_in: int, tokens_out: int, cache_read: int = 0,
                 cache_write: int = 0, cache_ttl: str | None = None) -> float | None:
        """USD for one request; None when the model is not in the table.
        `tokens_in` is the UNCACHED prompt remainder; the tier is chosen on the whole prompt.
        Cache reads/writes use the entry's `cache_read` / `cache_write` (5m) or `cache_write_1h`
        (cache_ttl="1h") prices; absent = the plain input price."""
        p = self.price(model)
        if p is None:
            return None
        prompt = tokens_in + cache_read + cache_write
        eff = dict(p)
        t = p.get("tier")
        if t and prompt > int(t["input_above_tokens"]):
            eff.update({k: v for k, v in t.items() if k != "input_above_tokens"})
        usd = (tokens_in * eff["in"] + tokens_out * eff["out"]
               + cache_read * eff.get("cache_read", eff["in"])
               + cache_write * (eff.get("cache_write_1h", eff.get("cache_write", eff["in"]))
                                if cache_ttl == "1h" else eff.get("cache_write", eff["in"]))) / 1e6
        return usd


def lookup_price(model: str, profile: "Profile | None" = None) -> tuple["Profile", dict] | None:
    """(profile, price entry) for a model id: the given profile first, then every shipped one."""
    names = ([profile.name] if profile else []) + [n for n in PROFILE_NAMES]
    seen = set()
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        p = profile if (profile and n == profile.name) else None
        try:
            p = p or load_profile(n)
        except ProviderError:
            continue
        e = p.price(model)
        if e:
            return p, e
    return None


def price_tuple(model: str, profile: "Profile | None" = None) -> tuple[float, float] | None:
    """(USD/MTok in, out) of a model id from the profile price tables (the ONE price source),
    or None when no shipped profile prices it. Prompts above a tier threshold are not
    represented here (use Profile.cost_usd for exact per-request cost)."""
    hit = lookup_price(model, profile)
    return (float(hit[1]["in"]), float(hit[1]["out"])) if hit else None


def thinking_soft_extra(profile: "Profile", mode: str = "disabled") -> dict:
    """`soft_extra_body` for chat(): Anthropic's endpoint otherwise runs adaptive thinking that
    can eat a short answer's whole max_tokens (stop=length, empty text). Only for the anthropic
    endpoint; a model that rejects the knob falls back to the default (see chat())."""
    host = (profile.base_url or "")
    if mode == "disabled" and (profile.name == "anthropic" or "api.anthropic.com" in host):
        return {"thinking": {"type": "disabled"}}
    return {}


def load_profile(name: str, profiles_dir: str | None = None, *, base_url: str | None = None,
                 key_env: str | None = None, pi_provider: str | None = None,
                 cache: bool | None = None) -> Profile:
    """Load profiles/<name>.json. Overrides (custom endpoint): base_url, key_env, pi_provider;
    cache=True/False forces prompt caching on/off (None = the profile's `cache.enabled`)."""
    d = profiles_dir or PROFILES_DIR
    path = os.path.join(d, f"{name}.json")
    if not os.path.isfile(path):
        have = sorted(f[:-5] for f in os.listdir(d)) if os.path.isdir(d) else []
        raise ProviderError(f"unknown profile {name!r} (have {have}; dir {d})")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for k in ("name", "key_env", "auth", "models", "pi", "prices"):
        if k not in data:
            raise ProviderError(f"profile {name}: missing key {k!r}")
    if data["name"] != name:
        raise ProviderError(f"profile file {name}.json declares name {data['name']!r}")
    if (data["auth"].get("style") or "") not in ("bearer", "x-api-key"):
        raise ProviderError(f"profile {name}: auth.style must be bearer|x-api-key")
    data = copy.deepcopy(data)
    if base_url:
        data["base_url"] = base_url
    if key_env:
        data["key_env_override"] = key_env
    if pi_provider:
        data["pi"]["provider"] = pi_provider
    if cache is not None:
        data.setdefault("cache", {})["enabled"] = bool(cache)
    return Profile(name, data)


def resolve_models(profile: Profile, overrides: dict | None = None) -> dict:
    """role -> model id: profile default, then per-role override (None/'' ignored)."""
    out = {r: profile.default_model(r) for r in ROLES}
    for r, m in (overrides or {}).items():
        if r not in ROLES:
            raise ProviderError(f"unknown role {r!r}; roles: {ROLES}")
        if m:
            out[r] = m
    return out


def require_models(models: dict, roles) -> None:
    missing = [r for r in roles if not models.get(r)]
    if missing:
        raise ProviderError(f"no model for role(s) {missing}: pass --model-<role> "
                            "(this profile has no default)")


# ----------------------------------------------------------------------- keys

def key_env_name(profile: Profile, key_env: str | None = None) -> str:
    return key_env or profile.data.get("key_env_override") or profile.key_env


def resolve_key(profile: Profile, key_env: str | None = None, environ=None) -> str:
    """The API key value (never print it). Raises ProviderError naming the VARIABLE only."""
    environ = os.environ if environ is None else environ
    name = key_env_name(profile, key_env)
    v = str(environ.get(name, "")).strip()
    if not v:
        raise ProviderError(f"{name} is not set (required by profile {profile.name})")
    return v


def key_problem(profile: Profile, key_env: str | None = None, environ=None) -> str | None:
    try:
        resolve_key(profile, key_env, environ)
        return None
    except ProviderError as e:
        return str(e)


def pi_env(profile: Profile, key_env: str | None = None, environ=None) -> dict:
    """Env additions for pi: the key exposed under pi's own variable for this provider
    (e.g. --key-env EVAL_API_KEY -> ANTHROPIC_API_KEY). Empty when no override is needed."""
    environ = os.environ if environ is None else environ
    native = profile.pi.get("key_env")
    if not native:
        return {}
    name = key_env_name(profile, key_env)
    if name == native:
        return {}
    return {native: resolve_key(profile, key_env, environ)}


# ------------------------------------------------------------------- families

_FAMILY_RULES = (   # (substring of the lowercased model id, family); first match wins
    ("claude", "anthropic"), ("anthropic", "anthropic"),
    ("gpt", "openai"), ("openai", "openai"),
    ("qwen", "qwen"), ("glm", "zai"), ("z-ai", "zai"), ("zai", "zai"),
    ("gemini", "google"), ("gemma", "google"), ("google", "google"),
    ("llama", "meta"), ("meta-llama", "meta"), ("mistral", "mistral"),
    ("deepseek", "deepseek"), ("nemotron", "nvidia"), ("grok", "xai"), ("x-ai", "xai"),
)


def model_family(model: str | None) -> str | None:
    """Vendor family of a model id (None = unknown). Route prefixes (`openrouter/`,
    `anthropic/`) are looked through only when the tail is not itself recognisable."""
    m = (model or "").strip().lower()
    if not m:
        return None
    tail = m.split("/")[-1]
    for needle, fam in _FAMILY_RULES:
        if needle in tail:
            return fam
    for needle, fam in _FAMILY_RULES:
        if needle in m:
            return fam
    return None


def norm_model(model: str | None) -> str:
    """Comparable model identity across routers: strip the vendor prefix, unify . and -."""
    m = (model or "").strip().lower().split("/")[-1]
    return m.replace(".", "-")


def family_conflict(agent_model: str, judge_model: str | None) -> str | None:
    """Error text when judge and agent share a family (both known), else None."""
    fa, fj = model_family(agent_model), model_family(judge_model)
    if fa and fj and fa == fj:
        return (f"judge model {judge_model!r} and agent model {agent_model!r} are the same "
                f"family ({fa}); the blind judge must be a different family from the agent")
    return None


def judge_family_verdict(agent_models, judge_model: str | None, allow_same_family: bool) -> dict:
    """Judge-independence decision.

    agent_models: every model whose answers/compactions the judge will see (agent, answer,
    summarizer). Default: same family -> refused. With allow_same_family: permitted ONLY when
    the judge MODEL differs from every one of them; then judge_same_family=True and a loud
    warning. Returns {ok, error, judge_same_family, reason, warning}.
    """
    agents = [m for m in dict.fromkeys(agent_models) if m]
    conflicts = [c for c in (family_conflict(m, judge_model) for m in agents) if c]
    out = {"ok": True, "error": None, "judge_same_family": False, "reason": None,
           "warning": None, "judge_model": judge_model, "agent_models": agents,
           "judge_family": model_family(judge_model),
           "agent_families": sorted({model_family(m) or "?" for m in agents})}
    if not conflicts:
        return out
    if not allow_same_family:
        out.update(ok=False, error=conflicts[0] + " (pass --allow-same-family-judge to accept "
                                   "a same-family, different-model judge, with a recorded limitation)")
        return out
    same_model = [m for m in agents if norm_model(m) == norm_model(judge_model)]
    if same_model:
        out.update(ok=False, error=f"--allow-same-family-judge still requires judge model != "
                                   f"agent model; {judge_model!r} equals {same_model[0]!r}")
        return out
    reason = (f"judge {judge_model!r} is in the same model family ({out['judge_family']}) as the "
              f"agent-side model(s) {agents}; different model, same family: self-preference / "
              "shared-bias risk is NOT controlled")
    out.update(judge_same_family=True, reason=reason,
               warning="!!! LOUD WARNING: SAME-FAMILY JUDGE (--allow-same-family-judge). " + reason +
                       ". Reported as a limitation of every result.")
    return out


# ---------------------------------------------------------------------- budget

class Budget:
    """Hard USD cap. charge() accumulates; check() raises BudgetExceeded at/over the cap."""

    def __init__(self, limit_usd: float | None):
        if limit_usd is None or not (float(limit_usd) > 0):
            raise ProviderError("a positive --budget-usd is required for every paid run")
        self.limit = float(limit_usd)
        self.spent = 0.0
        self.calls = 0
        self.unpriced_calls = 0
        self._lock = threading.Lock()

    def charge(self, usd: float | None) -> None:
        with self._lock:
            self.calls += 1
            if usd is None:
                self.unpriced_calls += 1
            else:
                self.spent += float(usd)

    def check(self) -> None:
        if self.spent >= self.limit:
            raise BudgetExceeded(f"budget ${self.limit:.4f} exhausted (spent ${self.spent:.4f} "
                                 f"over {self.calls} calls)")

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.spent)


def add_budget_arg(ap: argparse.ArgumentParser, required: bool = False) -> None:
    ap.add_argument("--budget-usd", type=float, default=None, required=required,
                    help="HARD USD cap for this paid script (required for any live run; "
                         "protocol Amendment C.1). The script stops when it is reached")


# ------------------------------------------------------------------ chat() API

@dataclasses.dataclass
class ChatResult:
    text: str
    usage: dict
    cost_usd: float | None
    cost_source: str          # reported | table | none
    model: str
    profile: str
    latency_s: float
    raw: dict | None = None
    finish_reason: str | None = None     # provider's finish_reason (stop | length | ...)
    cost_nocache_usd: float | None = None  # same tokens at plain input price (savings = this - cost_usd)
    cached: bool = False                 # request sent through the cache-enabled native path


def _urlopen_transport(req: urllib.request.Request, timeout: float) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            body = {"error": {"message": f"HTTP {e.code}"}}
        return e.code, body


def build_request(profile: Profile, role_model: str, messages: list, *, max_tokens: int,
                  temperature=None, top_p=None, key: str, extra_body: dict | None = None
                  ) -> tuple[str, dict, dict]:
    """(url, headers, body) of one chat/completions call under the profile's rules."""
    if not profile.base_url:
        raise ProviderError(f"profile {profile.name}: no base_url (pass --base-url)")
    url = profile.base_url.rstrip("/") + profile.data.get("chat_path", "/chat/completions")
    headers = {"Content-Type": "application/json", "User-Agent": "neurocompact-eval/3.0"}
    headers.update(profile.data["auth"].get("headers") or {})
    if profile.data["auth"]["style"] == "x-api-key":
        headers["x-api-key"] = key
    else:
        headers["Authorization"] = f"Bearer {key}"
    body = {"model": role_model, "messages": messages, "max_tokens": int(max_tokens)}
    body.update(profile.sampling_kwargs(role_model, temperature=temperature, top_p=top_p))
    if profile.data.get("usage_include"):
        body["usage"] = {"include": True}          # OpenRouter: reported cost in usage.cost
    if extra_body:
        body.update(extra_body)
    return url, headers, body


ANTHROPIC_VERSION = "2023-06-01"


def _msg_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(str(b.get("text") or "") for b in (content or []) if isinstance(b, dict))


def build_native_request(profile: Profile, model: str, messages: list, *, max_tokens: int, key: str,
                         extra_body: dict | None = None, cache_prefix_chars: int | None = None
                         ) -> tuple[str, dict, dict]:
    """(url, headers, body) of one Anthropic /v1/messages call WITH prompt caching.

    The model sees exactly the OpenAI-shaped `messages` text: system messages become the
    `system` field (last block carries cache_control), user/assistant messages keep their order
    and text, and the LAST user message is cut at `cache_prefix_chars` into two text blocks of
    the same concatenated text, the first carrying cache_control (the stable prefix: long
    context first, question last). Nothing is added, removed or reordered: caching is cost-only.
    """
    if not profile.base_url:
        raise ProviderError(f"profile {profile.name}: no base_url (pass --base-url)")
    url = profile.base_url.rstrip("/") + "/messages"
    headers = {"Content-Type": "application/json", "User-Agent": "neurocompact-eval/3.0",
               "x-api-key": key, "anthropic-version": profile.cache_cfg.get("anthropic_version")
               or ANTHROPIC_VERSION}
    cc = profile.cache_control()
    sys_txt = [_msg_text(m.get("content")) for m in messages if m.get("role") == "system"]
    sys_txt = [t for t in sys_txt if t]
    msgs = [{"role": m["role"], "content": _msg_text(m.get("content"))}
            for m in messages if m.get("role") in ("user", "assistant")]
    last_user = max((i for i, m in enumerate(msgs) if m["role"] == "user"), default=None)
    if last_user is not None:
        t = msgs[last_user]["content"]
        n = int(cache_prefix_chars or 0)
        if 0 < n < len(t):
            msgs[last_user]["content"] = [{"type": "text", "text": t[:n], "cache_control": cc},
                                          {"type": "text", "text": t[n:]}]
        elif n >= len(t) > 0:
            msgs[last_user]["content"] = [{"type": "text", "text": t, "cache_control": cc}]
    body: dict = {"model": model, "max_tokens": int(max_tokens), "messages": msgs}
    if sys_txt:
        blocks = [{"type": "text", "text": t} for t in sys_txt]
        blocks[-1]["cache_control"] = cc
        body["system"] = blocks
    if extra_body:
        body.update({k: v for k, v in extra_body.items() if k != "reasoning"})
    return url, headers, body


def parse_usage(raw: dict) -> dict:
    """prompt_tokens = the WHOLE prompt (cached or not) for both wire formats; cached_tokens =
    cache reads, cache_write_tokens = cache creation (Anthropic native only)."""
    u = raw.get("usage") or {}
    if "prompt_tokens" not in u and "input_tokens" in u:           # Anthropic native /v1/messages
        rd = int(u.get("cache_read_input_tokens") or 0)
        wr = int(u.get("cache_creation_input_tokens") or 0)
        return {"prompt_tokens": int(u.get("input_tokens") or 0) + rd + wr,
                "completion_tokens": int(u.get("output_tokens") or 0),
                "cached_tokens": rd, "cache_write_tokens": wr, "reported_cost": None}
    det = u.get("prompt_tokens_details") or {}
    cached = int(det.get("cached_tokens") or 0)
    pt = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
    return {"prompt_tokens": pt, "completion_tokens": int(u.get("completion_tokens")
                                                           or u.get("output_tokens") or 0),
            "cached_tokens": cached, "cache_write_tokens": 0,
            "reported_cost": u.get("cost") or u.get("total_cost")}


def chat(role: str, messages: list, *, profile: Profile, models: dict | None = None,
         model: str | None = None, max_tokens: int = 700, temperature=None, top_p=None,
         budget: Budget | None = None, key_env: str | None = None, environ=None,
         timeout: float = 180, retries: int = 3, extra_body: dict | None = None,
         soft_extra_body: dict | None = None, transport=None, sleep=time.sleep,
         cache: bool | None = None, cache_prefix_chars: int | None = None) -> ChatResult:
    """One chat/completions call for `role` through `profile`.

    Model = `model` arg, else models[role], else the profile default. Sampling params are
    filtered by the profile's policy. Cost = provider-reported when present (OpenRouter),
    else the profile price table; with a Budget an unpriced model without reported cost
    raises UnpricedModel (a hard cap cannot be enforced blind). Retries 429/5xx/network.
    `soft_extra_body` fields (e.g. anthropic `thinking`) are sent first; an HTTP 400 that names
    one of them drops exactly those fields and retries once (a model that refuses the knob).
    Prompt caching (`cache`, default = the profile's `cache.enabled`; anthropic profile only):
    the call goes to the native /v1/messages endpoint with cache_control on the system prompt and
    on the first `cache_prefix_chars` characters of the last user message; cost accounting then
    uses the cache read/write prices. Same text reaches the model; only the bill changes.
    """
    if role not in ROLES:
        raise ProviderError(f"unknown role {role!r}")
    mid = model or (models or {}).get(role) or profile.default_model(role)
    if not mid:
        raise ProviderError(f"no model for role {role!r} in profile {profile.name}")
    if budget is not None:
        budget.check()
    key = resolve_key(profile, key_env, environ)
    eb = dict(extra_body or {})
    eb.update(soft_extra_body or {})
    use_cache = profile.cache_enabled if cache is None else (bool(cache) and profile.cache_enabled)
    # only when there is a stable prefix to mark (a system prompt or an explicit prefix): a call
    # with nothing to cache stays on the plain endpoint, byte-identical to the uncached run
    use_cache = use_cache and bool(cache_prefix_chars or any(m.get("role") == "system" for m in messages))
    if use_cache:
        url, headers, body = build_native_request(profile, mid, messages, max_tokens=max_tokens,
                                                  key=key, extra_body=eb,
                                                  cache_prefix_chars=cache_prefix_chars)
    else:
        url, headers, body = build_request(profile, mid, messages, max_tokens=max_tokens,
                                           temperature=temperature, top_p=top_p, key=key,
                                           extra_body=eb)
    send = transport or _urlopen_transport
    t0 = time.time()
    last = ""
    soft_left = bool(soft_extra_body)
    attempt = -1
    while True:
        attempt += 1
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        try:
            status, raw = send(req, timeout)
        except Exception as e:  # noqa: BLE001  (network error)
            status, raw = 0, {"error": {"message": type(e).__name__}}
        if status == 200:
            break
        last = f"HTTP {status}: {str((raw.get('error') or {}).get('message') or raw)[:200]}"
        if (status == 400 and soft_left
                and any(k in last.lower() for k in (soft_extra_body or {}))):
            for k in soft_extra_body:
                body.pop(k, None)
            soft_left = False
            attempt -= 1                      # the knob retry does not consume a network retry
            continue
        if status not in (0, 408, 429, 500, 502, 503, 504) or attempt >= retries - 1:
            raise ProviderError(f"{profile.name}/{mid}: {last}")
        sleep(min(30, 4 * (attempt + 1)))
    if use_cache:
        text = "".join(str(b.get("text") or "") for b in (raw.get("content") or [])
                       if isinstance(b, dict) and b.get("type") == "text")
        finish = raw.get("stop_reason")
    else:
        choice = ((raw.get("choices") or [{}])[0].get("message") or {})
        text = choice.get("content") or ""
        finish = (raw.get("choices") or [{}])[0].get("finish_reason")
    u = parse_usage(raw)
    cached = u["cached_tokens"]
    written = u.get("cache_write_tokens", 0)
    ttl = str(profile.cache_cfg.get("ttl") or "5m")
    cost, src, nocache = None, "none", None
    if u["reported_cost"] is not None:
        cost, src = float(u["reported_cost"]), "reported"
    else:
        c = profile.cost_usd(mid, max(0, u["prompt_tokens"] - cached - written), u["completion_tokens"],
                             cache_read=cached, cache_write=written, cache_ttl=ttl)
        if c is not None:
            cost, src = c, "table"
            nocache = profile.cost_usd(mid, u["prompt_tokens"], u["completion_tokens"])
    if budget is not None:
        if cost is None:
            budget.charge(None)
            raise UnpricedModel(f"{profile.name}/{mid} has no price entry and the provider "
                                "reported no cost: cannot enforce --budget-usd")
        budget.charge(cost)
    return ChatResult(text=text, usage=u, cost_usd=cost, cost_source=src, model=mid,
                      profile=profile.name, latency_s=time.time() - t0, raw=None,
                      finish_reason=finish, cost_nocache_usd=nocache, cached=use_cache)


# -------------------------------------------------------------------- pi side

def pi_flags(profile: Profile, model: str) -> list[str]:
    """pi CLI args selecting the model. openrouter keeps its historical shape."""
    prov = profile.pi.get("provider")
    if not prov:
        raise ProviderError(f"profile {profile.name}: no pi provider (use --pi-provider)")
    if profile.pi.get("model_style") == "openrouter_prefixed":
        return ["--model", f"openrouter/{model}", "--provider", prov]
    return ["--provider", prov, "--model", model]


def runner_provider_table() -> dict:
    """run_chainreplay.PROVIDERS built from the profile files (name -> pi/nc settings)."""
    out = {}
    for n in PROFILE_NAMES:
        if n == "custom":
            continue
        try:
            p = load_profile(n)
        except ProviderError:
            continue
        out[n] = {"pi_flags": p.pi["provider"], "nc_provider": p.pi["provider"],
                  "key_env": p.pi["key_env"], "ctx_window": int(p.pi["ctx_window"])}
    return out


# ------------------------------------------------------------------- HARS env

def hars_env(profile: Profile, models: dict | None = None, key_env: str | None = None) -> list[str]:
    """Shell `export` lines that point the HARS server's extractor + keyword/query LLM at the
    profile. The key is referenced as "$VAR", never expanded here."""
    m = models or resolve_models(profile)
    require_models(m, ("extractor", "keyword"))
    kv = key_env_name(profile, key_env)
    h = profile.data.get("hars") or {}
    L = [f"# HARS server LLM env for profile '{profile.name}' (generated by providers.py hars-env)",
         f"# extractor={m['extractor']}  keyword/query={m['keyword']}",
         "# start the HARS MCP/HTTP server from a shell that has these exported, then reconnect /mcp"]
    if h.get("extractor_provider") == "openrouter":
        L += ["export HARS_MEMORY_EXTRACTOR_PROVIDER=openrouter",
              f"export HARS_MEMORY_OPENROUTER_MODEL={m['extractor']}",
              f"export HARS_MEMORY_OPENROUTER_BASE_URL={profile.base_url}",
              f"export HARS_MEMORY_EXTRACTOR_API_KEY_ENV={kv}"]
    else:
        # generic OpenAI-compatible endpoint = the server's 'local' provider
        L += ["export HARS_MEMORY_EXTRACTOR_PROVIDER=local",
              f"export HARS_MEMORY_EXTRACTOR_BASE_URL={profile.base_url}",
              f"export HARS_MEMORY_EXTRACTOR_MODEL={m['extractor']}"]
    L += [f"export HARS_MEMORY_QUERY_BASE_URL={profile.base_url}",
          f"export HARS_MEMORY_QUERY_MODEL={m['keyword']}",
          f'export HARS_MEMORY_LLM_API_KEY="${kv}"   # used by the query/keyword LLM (and the local extractor)']
    omit = any(profile.omits_sampling(m[r]) for r in ("extractor", "keyword"))
    if omit:
        L.append("export HARS_MEMORY_LLM_OMIT_SAMPLING_PARAMS=1   # Anthropic 5.x rejects temperature/top_p")
    else:
        L.append("# HARS_MEMORY_LLM_OMIT_SAMPLING_PARAMS not needed for this profile")
    L += ["# embedder is INDEPENDENT of the chat profile (e.g. Anthropic has no embeddings API): export your real",
          "# embedder here, and the run manifest records what the server reports (memory_status):",
          "#   export HARS_MEMORY_EMBED_MODEL=<model id>   [HARS_MEMORY_EMBED_PROVIDER=http HARS_MEMORY_EMBED_BASE_URL=... "
          "HARS_MEMORY_EMBED_HTTP_MODEL=...]"]
    if profile.name == "gemini":
        L.append("#   (optional, UNVERIFIED) Gemini embeddings over HTTP: HARS_MEMORY_EMBED_PROVIDER=http "
                 "HARS_MEMORY_EMBED_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai "
                 "HARS_MEMORY_EMBED_API_KEY=$GEMINI_API_KEY HARS_MEMORY_EMBED_HTTP_APPLY_PROMPTS=0; "
                 "model id/dimension not verified; changing the embedder needs a reindex")
    return L


# --------------------------------------------------------------- CLI helpers

def add_profile_args(ap: argparse.ArgumentParser, roles=ROLES) -> None:
    """--profile, --base-url, --pi-provider and --model-<role> for the given roles."""
    ap.add_argument("--profile", default=None, choices=list(PROFILE_NAMES),
                    help="provider profile (profiles/<name>.json): anthropic | openrouter | "
                         "gemini | custom (OpenAI-compatible --base-url). Frozen per decision "
                         "set (Amendment C.1)")
    ap.add_argument("--base-url", default=None,
                    help="custom OpenAI-compatible base URL (profile custom, or an override)")
    ap.add_argument("--pi-provider", default=None,
                    help="pi provider name for profile custom (must exist in pi's models.json)")
    ap.add_argument("--prompt-cache", choices=("auto", "on", "off"), default="auto",
                    help="provider prompt caching (cost-only, Amendment C.3): auto = the profile's "
                         "`cache.enabled` (ON for anthropic), on/off force it. Caching never changes "
                         "the text the model sees; every arm is cached identically")
    for r in roles:
        ap.add_argument(f"--model-{r.replace('_', '-')}", dest=f"model_{r}", default=None,
                        metavar="ID", help=f"model id for role '{r}' (default: the profile's)")


def cache_override(args) -> bool | None:
    """--prompt-cache auto|on|off -> None|True|False (for load_profile(cache=...))."""
    v = getattr(args, "prompt_cache", "auto")
    return None if v in (None, "auto") else v == "on"


def role_overrides(args, roles=ROLES) -> dict:
    return {r: getattr(args, f"model_{r}", None) for r in roles if getattr(args, f"model_{r}", None)}


# ------------------------------------------------------------------------ CLI

def _show(profile: Profile, models: dict) -> str:
    L = [f"profile: {profile.name} - {profile.data.get('description', '')}",
         f"base_url: {profile.base_url}  auth: {profile.data['auth']['style']}  key env: {profile.key_env}",
         f"sampling: {json.dumps(profile.data.get('sampling'))}",
         f"pi: provider={profile.pi.get('provider')} key_env={profile.pi.get('key_env')} "
         f"ctx_window={profile.pi.get('ctx_window')}", "", "role -> model (price in/out per Mtok, status)"]
    for r in ROLES:
        m = models.get(r)
        p = profile.price(m) if m else None
        L.append(f"  {r:13s} {str(m):32s} "
                 + (f"${p['in']}/${p['out']} {p['status']}" if p else "no price"))
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("show", "hars-env", "prices", "check-family"):
        s = sub.add_parser(name)
        add_profile_args(s)
        s.add_argument("--key-env", default=None)
        s.set_defaults(profile=None)
        if name == "check-family":
            s.add_argument("--allow-same-family-judge", action="store_true")
    a = ap.parse_args(argv)
    try:
        prof = load_profile(a.profile or "anthropic", base_url=a.base_url, key_env=a.key_env,
                            pi_provider=a.pi_provider)
        models = resolve_models(prof, role_overrides(a))
        if a.cmd == "show":
            print(_show(prof, models))
        elif a.cmd == "hars-env":
            print("\n".join(hars_env(prof, models, a.key_env)))
        elif a.cmd == "prices":
            for m, p in prof.data["prices"].items():
                print(f"{m}: in ${p['in']} out ${p['out']} /Mtok  [{p['status']}]  {p.get('source', '')}")
        else:
            v = judge_family_verdict([models["agent"], models["answer"], models["summarizer"]],
                                     models["judge"], a.allow_same_family_judge)
            print(json.dumps(v, indent=1))
            return 0 if v["ok"] else 2
    except ProviderError as e:
        print(f"!!! {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
