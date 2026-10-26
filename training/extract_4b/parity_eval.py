#!/usr/bin/env python3
"""Parity gate: student (merged QLoRA) vs 27B teacher on 20 held-out prompts.

Loads ./runs/qlora-v1/merged, greedily extracts on every parity_holdout prompt,
scores against the teacher reference outputs shipped in the SFT corpus:

  entity F1      normalized-name set match (primary); name+type strict (diag)
  relation F1    normalized (src,tgt) UNDIRECTED pair match (primary; the
                 system prompt explicitly says relationships are undirected),
                 directed match reported as diagnostic
  format validity  every line well-formed (entity=4 / relation=5 fields),
                 exactly one trailing <|COMPLETE|>, no truncation
  hallucination  student entities whose normalized name does not appear in
                 the source input text (same metric computed for the teacher
                 as baseline)
  think-leak     any "<think>" emitted (hard fail; killed a prior pipeline)

Writes parity_results.json + parity_predictions.jsonl next to the run dir.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "qlora-v1"
MERGED = RUN / "merged"
DATA = HERE / "data" / "parity_holdout.jsonl"
MAX_NEW_TOKENS = 2816
ALLOWED_TYPES = {"Concept", "Document", "Decision", "Event", "Person", "Tool", "Task", "Other"}
THINK_BLOCK = "<think>\n\n</think>\n\n"


def log(msg: str) -> None:
    print(f"[parity {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = s.replace("’", "'").replace("`", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,:;!" + '"')


def parse_records(text: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    entities, relations, bad = [], [], 0
    complete = any(ln.strip() == "<|COMPLETE|>" for ln in lines)
    complete_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "<|COMPLETE|>"), None)
    body = [ln for ln in lines if ln.strip() != "<|COMPLETE|>"]
    extra_after_complete = (
        [ln for ln in lines[complete_idx + 1:] if ln.strip()] if complete_idx is not None else []
    )
    for ln in body:
        parts = ln.split("<|#|>")
        kind = parts[0].strip()
        if kind == "entity" and len(parts) == 4:
            entities.append({"name": parts[1].strip(), "type": parts[2].strip(), "desc": parts[3].strip()})
        elif kind == "relation" and len(parts) == 5:
            relations.append({"src": parts[1].strip(), "tgt": parts[2].strip(),
                              "kw": parts[3].strip(), "desc": parts[4].strip()})
        else:
            bad += 1
    return {
        "entities": entities, "relations": relations, "bad_lines": bad,
        "complete": complete, "lines_after_complete": len(extra_after_complete),
    }


def entity_metrics(student: list[dict], ref: list[dict], source_text: str):
    sn = {norm(e["name"]) for e in student}
    rn = {norm(e["name"]) for e in ref}
    st = {(norm(e["name"]), e["type"]) for e in student}
    rt = {(norm(e["name"]), e["type"]) for e in ref}
    inter = sn & rn
    prec = len(inter) / len(sn) if sn else 0.0
    rec = len(inter) / len(rn) if rn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    it = st & rt
    pt = len(it) / len(st) if st else 0.0
    rt_ = len(it) / len(rt) if rt else 0.0
    f1t = 2 * pt * rt_ / (pt + rt_) if pt + rt_ else 0.0
    # hallucination: student entity name not present in source text
    src = norm(source_text)
    hall = [n for n in sn if n not in src]
    return {
        "n_student": len(sn), "n_ref": len(rn), "n_matched": len(inter),
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "f1_name_type": round(f1t, 4),
        "hallucinated": hall,
    }


def relation_metrics(student: list[dict], ref: list[dict]):
    def pairs(rels, directed):
        out = set()
        for r in rels:
            a, b = norm(r["src"]), norm(r["tgt"])
            out.add((a, b) if directed else (min(a, b), max(a, b)))
        return out

    res = {}
    for directed, key in [(True, "directed"), (False, "undirected")]:
        s, r = pairs(student, directed), pairs(ref, directed)
        inter = s & r
        p = len(inter) / len(s) if s else 0.0
        rc = len(inter) / len(r) if r else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        res[key] = {"n_student": len(s), "n_ref": len(r), "n_matched": len(inter),
                    "precision": round(p, 4), "recall": round(rc, 4), "f1": round(f1, 4)}
    return res


def main() -> None:
    rows = [json.loads(line) for line in DATA.open()]
    log(f"{len(rows)} parity prompts")

    tok = AutoTokenizer.from_pretrained(str(MERGED))
    model = AutoModelForImageTextToText.from_pretrained(
        str(MERGED), dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
    model.eval()

    # does the template prefill the empty think block in the generation prompt?
    probe = tok.apply_chat_template(
        [{"role": "system", "content": "S."}, {"role": "user", "content": "U."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefill = bool(re.search(r"<think>\s*</think>\s*\n*$", probe))
    log(f"generation prompt prefill-think={prefill}")
    pad_id = tok.pad_token_id or tok.eos_token_id

    preds = []
    t_start = time.time()
    for i, r in enumerate(rows):
        msgs = [{"role": "system", "content": r["system"]},
                {"role": "user", "content": r["user"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
        if not prefill:
            prompt = prompt + THINK_BLOCK
        ids = tok(prompt, return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 repetition_penalty=1.0, pad_token_id=pad_id)
        gen = out[0][ids["input_ids"].shape[1]:]
        dt = time.time() - t0
        text = tok.decode(gen, skip_special_tokens=True)
        preds.append({
            "id": r["id"], "prediction": text, "reference": r["output"],
            "source_chars": len(r["user"]),
            "gen_tokens": int(gen.shape[0]), "gen_seconds": round(dt, 1),
            "tokens_per_s": round(gen.shape[0] / dt, 1),
        })
        log(f"[{i+1}/{len(rows)}] {r['id']}: {gen.shape[0]} tok / {dt:.0f}s")
    total_s = time.time() - t_start

    # ---------------- scoring ----------------
    per = []
    think_leaks = 0
    format_ok_count = 0
    agg = {
        "ent_stud": 0, "ent_ref": 0, "ent_match": 0,
        "rel_stud": 0, "rel_ref": 0, "rel_match": 0,
        "stud_ents": [], "ref_ents": [], "hall": [],
    }
    for p in preds:
        s = parse_records(p["prediction"])
        t = parse_records(p["reference"])
        per.append({"id": p["id"], "student": s, "teacher": t})
        think = "<think>" in p["prediction"]
        think_leaks += think
        fmt_ok = (
            s["bad_lines"] == 0 and s["complete"] and s["lines_after_complete"] == 0
            and all(e["type"] in ALLOWED_TYPES for e in s["entities"])
        )
        format_ok_count += fmt_ok
        agg["ent_stud"] += len({norm(e['name']) for e in s['entities']})
        agg["ent_ref"] += len({norm(e['name']) for e in t['entities']})
        agg["ent_match"] += len({norm(e['name']) for e in s['entities']} & {norm(e['name']) for e in t['entities']})
        for e in s["entities"]:
            agg["stud_ents"].append(norm(e["name"]))
        for e in t["entities"]:
            agg["ref_ents"].append(norm(e["name"]))

    # relation aggregation (needs the parsed pairs)
    rel_pairs_s, rel_pairs_r = [], []
    for item in per:
        rel_pairs_s.append([(norm(r["src"]), norm(r["tgt"])) for r in item["student"]["relations"]])
        rel_pairs_r.append([(norm(r["src"]), norm(r["tgt"])) for r in item["teacher"]["relations"]])
    rel_s = {tuple(sorted(x)) for pr in rel_pairs_s for x in pr}
    rel_r = {tuple(sorted(x)) for pr in rel_pairs_r for x in pr}
    rel_i = rel_s & rel_r

    # hallucination vs source text: re-tokenize source from the data rows
    src_by_idx = {i: rows[i]["user"] for i in range(len(rows))}
    hall_list = []
    for i, item in enumerate(per):
        src = norm(src_by_idx[i])
        for name in {norm(e["name"]) for e in item["student"]["entities"]}:
            if name not in src:
                hall_list.append({"id": item["id"], "entity": name})

    p_e = agg["ent_match"] / agg["ent_stud"] if agg["ent_stud"] else 0
    r_e = agg["ent_match"] / agg["ent_ref"] if agg["ent_ref"] else 0
    f1_e = 2 * p_e * r_e / (p_e + r_e) if p_e + r_e else 0
    p_r = len(rel_i) / len(rel_s) if rel_s else 0
    r_r = len(rel_i) / len(rel_r) if rel_r else 0
    f1_r = 2 * p_r * r_r / (p_r + r_r) if p_r + r_r else 0

    n = len(preds)
    results = {
        "model": str(MERGED),
        "n_prompts": n,
        "micro_entity": {
            "precision": round(p_e, 4), "recall": round(r_e, 4), "f1": round(f1_e, 4),
            "n_student": agg["ent_stud"], "n_reference": agg["ent_ref"], "n_matched": agg["ent_match"],
        },
        "micro_relation_undirected": {
            "precision": round(p_r, 4), "recall": round(r_r, 4), "f1": round(f1_r, 4),
            "n_student": len(rel_s), "n_reference": len(rel_r), "n_matched": len(rel_i),
        },
        "format_valid_outputs": f"{format_ok_count}/{n}",
        "format_errors": n - format_ok_count,
        "think_leaks": think_leaks,
        "hallucination": {
            "n_student_entities": len(set(agg["stud_ents"])),
            "n_hallucinated": len({(h['id'], h['entity']) for h in hall_list}),
            "rate": round(len({(h['id'], h['entity']) for h in hall_list}) /
                          max(1, sum(len({norm(e['name']) for e in item['student']['entities']}) for item in per)), 4),
            "examples": hall_list[:25],
        },
        "throughput": {
            "total_gen_seconds": round(total_s, 1),
            "total_gen_tokens": sum(p["gen_tokens"] for p in preds),
            "mean_tokens_per_s": round(sum(p["tokens_per_s"] for p in preds) / n, 1),
        },
        "ship_gate": {},
    }
    results["ship_gate"] = {
        "entity_f1_ge_0.85": f1_e >= 0.85,
        "zero_format_errors": (n - format_ok_count) == 0,
        "hallucination_lt_0.05": results["hallucination"]["rate"] < 0.05,
        "no_think_leaks": think_leaks == 0,
        "PASS": (f1_e >= 0.85 and (n - format_ok_count) == 0
                 and results["hallucination"]["rate"] < 0.05 and think_leaks == 0),
    }

    (RUN / "parity_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    with (RUN / "parity_predictions.jsonl").open("w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log("RESULTS:\n" + json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
