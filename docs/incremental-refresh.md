# Incremental refresh (`hars_memory.scripts.incremental_refresh`)

`plan | apply | audit | repair-orphans` for a LightRAG index (JSON KV stores + graphml + Qdrant). Same environment as `memory-index`.

- `plan` (read-only, no LLM, no embedder): classifies every walked document against the fingerprint sidecar AND the content LightRAG actually stored (`kv_store_full_docs.json`, after LightRAG's sanitising): `new`, `changed`, `retry` (doc_status exists but is not `processed`), `adopt` (processed, no fingerprint, content matches: record only), `unchanged`, `gone` (fingerprinted, no longer walked), `kept` (protected by `--keep-ids-file`). GC is refused when the walk covers < 50 % of the fingerprinted ids.
- `apply`: an unchanged index is a strict no-op (does not even start LightRAG). Otherwise one batched deletion (LightRAG's own `adelete_by_doc_id` per document, but the entity/relation rebuild and the storage flushes are deferred and run once for the union), then inserts in verified batches; fingerprints are persisted after each verified batch.
- `audit`: KV stores vs graph vs Qdrant (`--prefixes "" pier_kb` also compares the `pier_kb_lightrag_vdb_*` copies). `repair-orphans` removes Qdrant points that have no KV/graph counterpart (left behind by a killed run).
- `HARS_MEMORY_ID_PATH_MAP=ACTUAL=LOGICAL` (`ingest/document.py`): doc ids are `sha256(absolute path)`; the map lets a refresh run from another checkout location reproduce the ids stored in the index.

Why not `memory-index --refresh-changed`: it passes ALL documents to `ainsert` (one `dup-...` doc_status record per document per run), deletes with one full rebuild + two full flushes per document (60-180 s each on the Pier index), and saves fingerprints only at the end of the run.

## Usage

```bash
python -m hars_memory.scripts.incremental_refresh plan  --paths kb docs/x.md [--keep-ids-file ids.txt] [--no-gc] [--trust-sidecar] [--json plan.json]
python -m hars_memory.scripts.incremental_refresh apply --paths kb docs/x.md [--batch-size 25] [--dry-run] [--report report.json]
python -m hars_memory.scripts.incremental_refresh audit [--prefixes "" pier_kb] [--json audit.json]
python -m hars_memory.scripts.incremental_refresh repair-orphans [--prefix pier_kb] [--apply]
```

- `--paths` must be the FULL set of ingest roots, never a subset: a document that is fingerprinted but not walked counts as `gone` and is deleted (the 50 % guard only catches gross narrowing). Use `--no-gc` or `--keep-ids-file` when in doubt.
- `--trust-sidecar` classifies by the fingerprint sidecar only and skips the comparison with the stored `full_docs` content.
- `apply --dry-run` prints the plan summary and exits without starting LightRAG.
- Exit codes: `plan` 0; `apply` 0 when every deletion and insert succeeded, 1 otherwise, 130 when interrupted (re-run to resume; verified batches are never repeated); `audit` 0 when all checks pass, 1 otherwise; `repair-orphans` is a dry run unless `--apply` is given.
- Environment: `HARS_MEMORY_INDEX_DIR`, `HARS_MEMORY_QDRANT_URL` (the audit default is `http://localhost:6335`), `HARS_MEMORY_QDRANT_COLLECTION` (workspace id), `HARS_MEMORY_FINGERPRINT_STORE` (optional), plus the extractor and embedder settings from [configuration](configuration.md). `apply` honours `HARS_MEMORY_EXTRACTOR_PROVIDER`, so a hosted extractor works without a GPU.

## Audit notes

LightRAG derives a relationship's Qdrant point id from `md5(src + tgt)`. Two different edges whose names concatenate to the same string share one point id, so one of them has no vector of its own. `audit` does not fail on this: such edges are listed under `relationship_id_collisions` and the relationships check reports them as `(+N edge(s) sharing a point id ...)`. Self-loop edges are tolerated by this lookup.
