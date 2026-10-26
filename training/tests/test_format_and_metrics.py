"""Tests for extraction output parsing and parity metrics.

Pure stdlib: runs in the repo venv without torch/transformers, as part of
`uv run pytest -q`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from common import (
    COMPLETE,
    DELIM,
    entity_f1_metrics,
    evaluate_pair,
    extract_source_text,
    hallucination_rate,
    parse_extraction_output,
    relation_f1_metrics,
)


def _ent(name, typ, desc="d"):
    return f"entity{DELIM}{name}{DELIM}{typ}{DELIM}{desc}"


def _rel(src, tgt, kw="k", desc="d"):
    return f"relation{DELIM}{src}{DELIM}{tgt}{DELIM}{kw}{DELIM}{desc}"


class TestParse:
    def test_entities_and_relations(self):
        text = "\n".join([_ent("embassy task", "Task", "runs control loops"), _rel("embassy task", "FOC task", "control, motion", "feeds targets"), COMPLETE])
        p = parse_extraction_output(text)
        assert p.complete is True
        assert p.format_errors == []
        assert p.entities == [("embassy task", "Task", "runs control loops")]
        assert p.relations == [("embassy task", "FOC task", "control, motion", "feeds targets")]

    def test_complete_optional(self):
        p = parse_extraction_output(_ent("a", "Concept"))
        assert p.complete is False and len(p.entities) == 1

    def test_malformed_field_counts(self):
        bad_entity = f"entity{DELIM}only-name"
        bad_rel = f"relation{DELIM}a{DELIM}b{DELIM}only-three"
        p = parse_extraction_output("\n".join([bad_entity, bad_rel]))
        assert len(p.format_errors) == 2
        assert p.entities == [] and p.relations == []

    def test_empty_fields_are_errors(self):
        p = parse_extraction_output(f"entity{DELIM}name{DELIM}{DELIM}desc")
        assert len(p.format_errors) == 1

    def test_noise_lines_ignored(self):
        text = "\n".join(["</Output>", "var x=1;", "```", _ent("ok", "Tool")])
        p = parse_extraction_output(text)
        assert len(p.entities) == 1
        assert p.format_errors == []  # non-record noise is skipped, not an error

    def test_delim_garbage_is_format_error(self):
        p = parse_extraction_output(f"{DELIM}SESSION_x.md{DELIM}file.scss{DELIM}tags{DELIM}desc")
        assert len(p.format_errors) == 1


class TestMetrics:
    def test_perfect_match(self):
        gold = [("embassy task", "Task"), ("FOC task", "Task")]
        p, r, f = entity_f1_metrics(gold, gold)
        assert (p, r, f) == (1.0, 1.0, 1.0)

    def test_name_and_type_both_must_match(self):
        gold = [("embassy task", "Task")]
        pred = [("embassy task", "Concept")]
        p, r, f = entity_f1_metrics(gold, pred)
        assert f == 0.0

    def test_partial(self):
        gold = [("a", "T"), ("b", "T"), ("c", "T")]
        pred = [("a", "T"), ("b", "T"), ("d", "T")]
        p, r, f = entity_f1_metrics(gold, pred)
        assert abs(p - 2 / 3) < 1e-9
        assert abs(r - 2 / 3) < 1e-9
        assert abs(f - 2 / 3) < 1e-9

    def test_case_and_whitespace_normalized(self):
        gold = [("Embassy  Task", "Task")]
        pred = [("embassy task", "Task")]
        _, _, f = entity_f1_metrics(gold, pred)
        assert f == 1.0

    def test_multiset_not_set(self):
        gold = [("dup", "T")]
        pred = [("dup", "T"), ("dup", "T")]
        p, _, _ = entity_f1_metrics(gold, pred)
        assert p == 0.5  # extra duplicate counts against precision

    def test_relation_directional(self):
        gold = [("a", "b"), ("b", "a")]
        pred = [("a", "b")]
        p, r, f = relation_f1_metrics(gold, pred)
        assert p == 1.0 and r == 0.5

    def test_relation_ignores_desc(self):
        gold = [("a", "b", "kw1", "some desc")]
        pred = [("a", "b", "kw2", "other desc")]
        _, _, f = relation_f1_metrics(gold, pred)
        assert f == 1.0

    def test_empty_pred(self):
        p, r, f = entity_f1_metrics([("a", "T")], [])
        assert (p, r, f) == (0.0, 0.0, 0.0)
        assert entity_f1_metrics([], [])[2] == 0.0


class TestHallucination:
    def test_all_in_source(self):
        src = "The priority embassy task runs cascaded P and PI control loops."
        ents = [("priority embassy task", "Task"), ("P and PI control loops", "Concept")]
        assert hallucination_rate(ents, src) == 0.0

    def test_hallucinated_entity(self):
        src = "The embassy task runs control loops."
        ents = [("embassy task", "Task"), ("quantum flux capacitor", "Tool")]
        assert abs(hallucination_rate(ents, src) - 0.5) < 1e-9

    def test_case_insensitive(self):
        src = "The Embassy Task runs."
        assert hallucination_rate([("embassy task", "Task")], src) == 0.0

    def test_empty_entities(self):
        assert hallucination_rate([], "src") == 0.0


class TestEvaluatePair:
    def test_full_pair(self):
        user = "---Data---\n<Input Text>\n```\nThe embassy task runs the FOC task.\n```\n"
        gold = "\n".join([_ent("embassy task", "Task"), _rel("embassy task", "FOC task"), COMPLETE])
        m = evaluate_pair(gold, gold, user)
        assert m["entity_f1"] == 1.0
        assert m["relation_f1"] == 1.0
        assert m["hallucination_rate"] == 0.0
        assert m["n_format_errors"] == 0
        assert m["complete"] is True

    def test_format_errors_surface(self):
        m = evaluate_pair(_ent("a", "T"), f"entity{DELIM}broken")
        assert m["n_format_errors"] == 1

    def test_source_extraction_fallback(self):
        assert extract_source_text("no marker here") == "no marker here"
        src = extract_source_text("x<Input Text>\n```\nreal source\n```\n")
        assert "real source" in src


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
