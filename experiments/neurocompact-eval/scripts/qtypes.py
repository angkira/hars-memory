#!/usr/bin/env python3
"""Question-type vocabulary and the Amendment C.2 relabel (analysis-time mapping only).

Amendment C.2 (REPORT-PROTOCOL.md 4a.2): the OpenHands items historically stored as
``qtype = "multihop"`` are a LOCATOR in an old span + a FACT in a newer span (the dataset
agent's spot check: only 1 of 10 genuinely needs both hops), NOT cross-compaction reasoning.
They are therefore analysed as ``xspan_locator``. The data files are NOT rewritten and the
runners keep writing the stored label; every analysis script maps at LOAD time through
``canon_qtype`` so old and new artifacts stay byte-compatible. Genuine multi-hop is BEAM's
``multi_session_reasoning`` (``source_qtype`` of the long-session records).

    canon_qtype("multihop")                                  -> "xspan_locator"
    canon_qtype("multihop", "multi_session_reasoning")       -> "multi_session_reasoning"
    canon_qtype("single", "information_extraction")          -> "information_extraction"
    canon_qtype("single")                                    -> "single"
    canon_qtype("multihop", legacy=True)                     -> "multihop"   (pre-C.2 behaviour)
"""
from __future__ import annotations

XSPAN_LOCATOR = "xspan_locator"
LEGACY_MULTIHOP = ("multihop", "multi_hop", "multi-hop", "cross_compaction",
                   "cross-compaction", "xcompaction")
BEAM_TYPES = ("information_extraction", "knowledge_update", "temporal_reasoning",
              "multi_session_reasoning", "contradiction_resolution", "abstention",
              "event_ordering", "instruction_following", "preference_following", "summarization")
# C.2: the declared long-session strata (genuine multi-hop first)
GENUINE_MULTIHOP = ("multi_session_reasoning",)
DECLARED_STRATA = ("multi_session_reasoning", "knowledge_update", "temporal_reasoning",
                   "contradiction_resolution")


def canon_qtype(qtype, source_qtype=None, legacy: bool = False):
    """Analysis label of a question (see module doc). None stays None."""
    src = str(source_qtype).strip().lower() if source_qtype else ""
    q = str(qtype).strip().lower() if qtype else ""
    if legacy:
        return q or None
    if src in BEAM_TYPES:
        return src
    if q in LEGACY_MULTIHOP:
        return XSPAN_LOCATOR
    return q or None
