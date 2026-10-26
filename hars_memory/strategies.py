"""Validated, serializable indexing and search strategy contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping

import yaml

Scalar = str | int | float | bool

_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
_INDEX_OPTIONS: Final[dict[str, dict[str, type | tuple[type, ...]]]] = {
    "corpus": {
        "chunk_size": int,
        "chunk_overlap": int,
    },
    "lightrag": {
        "chunk_token_size": int,
        "chunk_overlap_tokens": int,
        "max_extract_input_tokens": int,
        "extractor_max_tokens": int,
        "max_parallel_insert": int,
        "llm_max_async": int,
        "max_gleaning": int,
        "extraction_language": str,
        "extractor_model": str,
        "embed_model": str,
        "vector_storage": str,
        "graph_storage": str,
    },
}
_LIGHTRAG_ENV: Final[dict[str, str]] = {
    "chunk_token_size": "HARS_MEMORY_CHUNK_TOKEN_SIZE",
    "chunk_overlap_tokens": "HARS_MEMORY_CHUNK_OVERLAP_TOKENS",
    "max_extract_input_tokens": "HARS_MEMORY_LLM_MAX_TOKEN_SIZE",
    "extractor_max_tokens": "HARS_MEMORY_EXTRACTOR_MAX_TOKENS",
    "max_parallel_insert": "HARS_MEMORY_MAX_PARALLEL_INSERT",
    "llm_max_async": "HARS_MEMORY_LLM_MAX_ASYNC",
    "max_gleaning": "HARS_MEMORY_MAX_GLEANING",
    "extraction_language": "HARS_MEMORY_EXTRACTION_LANGUAGE",
    "extractor_model": "HARS_MEMORY_EXTRACTOR_MODEL",
    "embed_model": "HARS_MEMORY_EMBED_MODEL",
    "vector_storage": "HARS_MEMORY_VECTOR_STORAGE",
    "graph_storage": "HARS_MEMORY_GRAPH_STORAGE",
}
_SEARCH_MODES: Final[dict[str, frozenset[str]]] = {
    "corpus": frozenset({"sparse", "dense", "fusion"}),
    "lightrag": frozenset(
        {"dense_only", "hybrid_bm25", "hybrid_bm25_rerank", "naive", "local", "global", "hybrid"}
    ),
}


class StrategyConfigurationError(ValueError):
    """A strategy or matrix is unsafe, unknown, or internally inconsistent."""


def _validate_name(name: str, label: str) -> str:
    if not _NAME_RE.fullmatch(name):
        raise StrategyConfigurationError(
            f"{label} must match {_NAME_RE.pattern!r}; got {name!r}"
        )
    return name


def _plain_options(options: Mapping[str, Scalar] | None) -> dict[str, Scalar]:
    return dict(options or {})


def _strategy_hash(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class IndexStrategy:
    name: str
    engine: str
    options: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_name(self.name, "index strategy name")
        if self.engine not in _INDEX_OPTIONS:
            raise StrategyConfigurationError(f"unknown index engine: {self.engine!r}")
        values = _plain_options(self.options)
        allowed = _INDEX_OPTIONS[self.engine]
        unknown = sorted(set(values) - set(allowed))
        if unknown:
            raise StrategyConfigurationError(
                f"unknown {self.engine} index option(s): {', '.join(unknown)}"
            )
        for key, value in values.items():
            expected = allowed[key]
            if isinstance(value, bool) or not isinstance(value, expected):
                raise StrategyConfigurationError(
                    f"index option {key!r} must be {getattr(expected, '__name__', expected)}"
                )
            if isinstance(value, int) and value < 0:
                raise StrategyConfigurationError(f"index option {key!r} must be non-negative")
        if self.engine == "corpus":
            chunk_size = int(values.get("chunk_size", 0))
            overlap = int(values.get("chunk_overlap", 0))
            if chunk_size and overlap >= chunk_size:
                raise StrategyConfigurationError("chunk_overlap must be smaller than chunk_size")
        else:
            chunk_size = int(values.get("chunk_token_size", 0))
            overlap = int(values.get("chunk_overlap_tokens", 0))
            if chunk_size and overlap >= chunk_size:
                raise StrategyConfigurationError(
                    "chunk_overlap_tokens must be smaller than chunk_token_size"
                )
            for key in ("max_parallel_insert", "llm_max_async"):
                if key in values and int(values[key]) < 1:
                    raise StrategyConfigurationError(f"index option {key!r} must be positive")
        object.__setattr__(self, "options", MappingProxyType(values))

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "engine": self.engine, "options": dict(self.options)}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> IndexStrategy:
        unknown = sorted(set(value) - {"name", "engine", "options"})
        if unknown:
            raise StrategyConfigurationError(
                f"unknown index strategy field(s): {', '.join(unknown)}"
            )
        options = value.get("options", {})
        if not isinstance(options, Mapping):
            raise StrategyConfigurationError("index strategy options must be an object")
        return cls(
            name=str(value.get("name", "default")),
            engine=str(value.get("engine", "")),
            options=options,  # type: ignore[arg-type]
        )

    @property
    def fingerprint(self) -> str:
        return _strategy_hash(self.to_dict())

    def environment(self) -> dict[str, str]:
        if self.engine != "lightrag":
            return {}
        return {_LIGHTRAG_ENV[key]: str(value) for key, value in self.options.items()}


@dataclass(frozen=True, slots=True)
class SearchStrategy:
    name: str
    backend: str
    mode: str
    top_k: int = 10
    alpha: float = 0.5
    pool_multiplier: int = 3
    chunk_top_k: int | None = None
    max_entity_tokens: int | None = None
    max_relation_tokens: int | None = None
    max_total_tokens: int | None = None
    embed_model: str | None = None
    rerank_model: str | None = None
    context_only: bool = True

    def __post_init__(self) -> None:
        _validate_name(self.name, "search strategy name")
        if self.backend not in _SEARCH_MODES:
            raise StrategyConfigurationError(f"unknown search backend: {self.backend!r}")
        if self.mode not in _SEARCH_MODES[self.backend]:
            raise StrategyConfigurationError(
                f"mode {self.mode!r} is not valid for backend {self.backend!r}"
            )
        if self.top_k < 1 or self.pool_multiplier < 1:
            raise StrategyConfigurationError("top_k and pool_multiplier must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise StrategyConfigurationError("alpha must be between 0 and 1")
        for name in (
            "chunk_top_k",
            "max_entity_tokens",
            "max_relation_tokens",
            "max_total_tokens",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise StrategyConfigurationError(f"{name} must be non-negative")
        for name in ("embed_model", "rerank_model"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise StrategyConfigurationError(f"{name} must not be blank")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "backend": self.backend,
            "mode": self.mode,
            "top_k": self.top_k,
            "alpha": self.alpha,
            "pool_multiplier": self.pool_multiplier,
            "chunk_top_k": self.chunk_top_k,
            "max_entity_tokens": self.max_entity_tokens,
            "max_relation_tokens": self.max_relation_tokens,
            "max_total_tokens": self.max_total_tokens,
            "embed_model": self.embed_model,
            "rerank_model": self.rerank_model,
            "context_only": self.context_only,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SearchStrategy:
        allowed = {
            "name",
            "backend",
            "mode",
            "top_k",
            "alpha",
            "pool_multiplier",
            "chunk_top_k",
            "max_entity_tokens",
            "max_relation_tokens",
            "max_total_tokens",
            "embed_model",
            "rerank_model",
            "context_only",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise StrategyConfigurationError(
                f"unknown search strategy field(s): {', '.join(unknown)}"
            )
        for name in (
            "top_k",
            "pool_multiplier",
            "chunk_top_k",
            "max_entity_tokens",
            "max_relation_tokens",
            "max_total_tokens",
        ):
            item = value.get(name)
            if item is not None and (isinstance(item, bool) or not isinstance(item, int)):
                raise StrategyConfigurationError(f"search field {name!r} must be an integer")
        alpha = value.get("alpha")
        if alpha is not None and (isinstance(alpha, bool) or not isinstance(alpha, (int, float))):
            raise StrategyConfigurationError("search field 'alpha' must be numeric")
        context_only = value.get("context_only")
        if context_only is not None and not isinstance(context_only, bool):
            raise StrategyConfigurationError("search field 'context_only' must be boolean")
        return cls(**dict(value))  # type: ignore[arg-type]

    @property
    def fingerprint(self) -> str:
        return _strategy_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class StrategyMatrix:
    name: str
    corpus_paths: tuple[Path, ...]
    queries_path: Path
    output_dir: Path
    index_strategies: tuple[IndexStrategy, ...]
    search_strategies: tuple[SearchStrategy, ...]
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    repetitions: int = 1
    warmups: int = 0
    primary_metric: str = "ndcg@10"
    primary_metric_direction: str = "higher"

    def __post_init__(self) -> None:
        _validate_name(self.name, "matrix name")
        if not self.corpus_paths or not self.index_strategies or not self.search_strategies:
            raise StrategyConfigurationError(
                "matrix requires corpus_paths, index_strategies, and search_strategies"
            )
        if self.repetitions < 1 or self.warmups < 0:
            raise StrategyConfigurationError("repetitions must be positive and warmups non-negative")
        if not self.k_values or any(k < 1 for k in self.k_values):
            raise StrategyConfigurationError("k_values must contain positive integers")
        if self.primary_metric_direction not in {"higher", "lower"}:
            raise StrategyConfigurationError(
                "primary_metric_direction must be 'higher' or 'lower'"
            )
        supported_metrics = {
            "mrr",
            "supersession_error_rate",
            "no_answer_hit_rate",
            *(f"recall@{k}" for k in self.k_values),
            *(f"ndcg@{k}" for k in self.k_values),
        }
        if self.primary_metric not in supported_metrics:
            raise StrategyConfigurationError(
                f"unsupported primary_metric {self.primary_metric!r}"
            )
        if self.primary_metric.startswith(("recall@", "ndcg@")):
            try:
                metric_k = int(self.primary_metric.partition("@")[2])
            except ValueError as exc:
                raise StrategyConfigurationError("invalid primary_metric cutoff") from exc
            if metric_k not in self.k_values:
                raise StrategyConfigurationError(
                    f"primary_metric {self.primary_metric!r} is absent from k_values"
                )
        names = [s.name for s in self.index_strategies]
        if len(names) != len(set(names)):
            raise StrategyConfigurationError("index strategy names must be unique")
        names = [s.name for s in self.search_strategies]
        if len(names) != len(set(names)):
            raise StrategyConfigurationError("search strategy names must be unique")
        engines = {s.engine for s in self.index_strategies}
        for search in self.search_strategies:
            if search.backend not in engines:
                raise StrategyConfigurationError(
                    f"search strategy {search.name!r} has no matching index backend"
                )

    def combinations(self) -> tuple[tuple[IndexStrategy, SearchStrategy], ...]:
        return tuple(
            (index, search)
            for index in self.index_strategies
            for search in self.search_strategies
            if index.engine == search.backend
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "name": self.name,
            "corpus_paths": [str(path) for path in self.corpus_paths],
            "queries_path": str(self.queries_path),
            "output_dir": str(self.output_dir),
            "index_strategies": [strategy.to_dict() for strategy in self.index_strategies],
            "search_strategies": [strategy.to_dict() for strategy in self.search_strategies],
            "k_values": list(self.k_values),
            "repetitions": self.repetitions,
            "warmups": self.warmups,
            "primary_metric": self.primary_metric,
            "primary_metric_direction": self.primary_metric_direction,
        }

    @property
    def fingerprint(self) -> str:
        return _strategy_hash(self.to_dict())


def load_strategy_matrix(path: Path) -> StrategyMatrix:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise StrategyConfigurationError("strategy matrix must be a YAML object")
    allowed = {
        "version",
        "name",
        "corpus_paths",
        "queries_path",
        "output_dir",
        "index_strategies",
        "search_strategies",
        "k_values",
        "repetitions",
        "warmups",
        "primary_metric",
        "primary_metric_direction",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise StrategyConfigurationError(f"unknown matrix field(s): {', '.join(unknown)}")
    if raw.get("version") != 1:
        raise StrategyConfigurationError("strategy matrix version must be 1")
    base = config_path.parent

    def resolve(value: object) -> Path:
        candidate = Path(str(value))
        return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

    index_raw = raw.get("index_strategies")
    search_raw = raw.get("search_strategies")
    corpus_raw = raw.get("corpus_paths")
    if not isinstance(index_raw, list) or not isinstance(search_raw, list) or not isinstance(corpus_raw, list):
        raise StrategyConfigurationError("strategy lists and corpus_paths must be YAML lists")
    if not all(isinstance(value, Mapping) for value in (*index_raw, *search_raw)):
        raise StrategyConfigurationError("every strategy entry must be a YAML object")
    k_values_raw = raw.get("k_values", (1, 3, 5, 10))
    if not isinstance(k_values_raw, (list, tuple)):
        raise StrategyConfigurationError("k_values must be a YAML list")
    return StrategyMatrix(
        name=str(raw.get("name", "")),
        corpus_paths=tuple(resolve(value) for value in corpus_raw),
        queries_path=resolve(raw.get("queries_path", "")),
        output_dir=resolve(raw.get("output_dir", "strategy-bench-results")),
        index_strategies=tuple(IndexStrategy.from_dict(value) for value in index_raw),
        search_strategies=tuple(SearchStrategy.from_dict(value) for value in search_raw),
        k_values=tuple(int(value) for value in k_values_raw),
        repetitions=int(raw.get("repetitions", 1)),
        warmups=int(raw.get("warmups", 0)),
        primary_metric=str(raw.get("primary_metric", "ndcg@10")),
        primary_metric_direction=str(raw.get("primary_metric_direction", "higher")),
    )


__all__ = [
    "IndexStrategy",
    "Scalar",
    "SearchStrategy",
    "StrategyConfigurationError",
    "StrategyMatrix",
    "load_strategy_matrix",
]
