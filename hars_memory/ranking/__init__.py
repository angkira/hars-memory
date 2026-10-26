"""Learned channel-fusion ranker (learning-to-rank) for the recall path.

Replaces the hand-tuned additive ``alpha*dense + (1-alpha)*sparse`` blend with a
small model (LightGBM LambdaRank; a tiny MLP is trained as the comparison) over
per-candidate rank-time features. The model is a pure function of channel
scores and cheap corpus metadata, so scoring is sub-millisecond on CPU and
needs no GPU, no LLM and no network.

Public surface:

    hars_memory.ranking.features    -- LTR_FEATURE_NAMES, extract_pool_features
    hars_memory.ranking.artifact    -- LtrRankerModel (manifest-guarded loader)
    hars_memory.ranking.rerank      -- LtrRanker / apply_ltr_ranker (fail-soft)
    hars_memory.ranking.train       -- training entry point (script helper)

The on-disk contract: a JSON manifest carrying ``feature_names`` (must equal
:data:`~hars_memory.ranking.features.LTR_FEATURE_NAMES` exactly) plus a model
file. An artifact trained on a different feature list is REFUSED, never
silently scored with shifted columns.
"""

from __future__ import annotations

from hars_memory.ranking.artifact import (
    LTR_MANIFEST_VERSION,
    LtrRankerModel,
)
from hars_memory.ranking.features import (
    LTR_FEATURE_NAMES,
    LtrCorpusContext,
    LtrFeatureRow,
    extract_pool_features,
)
from hars_memory.ranking.rerank import (
    apply_ltr_ranker,
    load_default_ranker,
)

__all__ = [
    "LTR_FEATURE_NAMES",
    "LTR_MANIFEST_VERSION",
    "LtrCorpusContext",
    "LtrFeatureRow",
    "LtrRankerModel",
    "apply_ltr_ranker",
    "extract_pool_features",
    "load_default_ranker",
]
