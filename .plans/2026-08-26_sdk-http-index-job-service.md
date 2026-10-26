# SDK + HTTP index-job service

Status: complete (implemented and verified 2026-08-26)

## Goal

Add a cloud-deployable service and Python SDK around the existing standalone
indexing package. A client uploads text files, creates or extends an index,
polls a durable job, and retrieves an immutable index version/artifact. The
same API must run locally or in the cloud, while GPU/LLM work stays inside a
server-side worker.

## Acceptance contract

- `POST /v1/index-jobs` accepts `.md`, `.txt`, `.json`, and `.py` files as
  multipart uploads and returns `202` with a durable job ID.
- `operation=create|extend`; extend targets an existing index and produces a
  new immutable version. The last successful version remains readable if a
  job fails or is cancelled.
- `engine=corpus` runs the existing CPU/no-LLM builder and is the E2E/CI path.
  `engine=lightrag` invokes the existing graph indexer in a worker process and
  is the remote-GPU production path.
- Job/index metadata uses SQLAlchemy with a configurable database URL: SQLite
  locally, PostgreSQL in cloud deployments.
- Input/output artifacts use a configurable store: local filesystem by
  default, S3-compatible object storage optionally.
- Authentication is fail-closed by default. Server-side API-key mappings
  identify tenants; job, index, and artifact reads are tenant-scoped.
- Idempotency keys deduplicate submissions per tenant.
- API returns immutable index descriptors, checksums, document/chunk counts,
  and an artifact download endpoint. A Qdrant-backed graph artifact explicitly
  reports its external vector workspace; the tarball never pretends to contain
  remote Qdrant vectors.
- A Python SDK covers submit/create/extend, get/wait/cancel, index inspection,
  and artifact download with checksum verification.
- Existing CLI and MCP behavior stays compatible.

## Implementation

1. Add optional service dependencies and console entrypoint in `pyproject.toml`.
2. Add `hars_memory/service/`:
   - typed API/domain models;
   - SQLAlchemy job/index-version repository;
   - local and S3-compatible artifact stores;
   - corpus and LightRAG engine adapters;
   - durable background worker with queued-job recovery and per-index locking;
   - FastAPI app, auth, validation, upload bounds, status/cancel/download routes.
3. Add `hars_memory/sdk.py`, a synchronous `httpx` client with typed results,
   bounded polling, and verified downloads.
4. Add subprocess E2E tests:
   - installed CLI build/status/query/incremental flow with dead LLM endpoints;
   - real MCP JSON-RPC-over-stdio flow with network attempts blocked;
   - real HTTP server + SDK flow using SQLite/local artifacts/corpus engine:
     create, wait, download, extend, idempotency, tenant isolation, failure
     preservation, and restart-visible state.
5. Add unit tests for PostgreSQL URL acceptance, artifact safety, state
   transitions, auth, upload validation, and LightRAG command construction.
6. Update README/config examples and record the session. Run focused E2E,
   then the complete suite and `git diff --check`.

## Safety boundaries

- E2E never selects `engine=lightrag`, loads an embedding model, contacts an
  LLM endpoint, mutates Cortex's live index, or uses a GPU.
- Uploaded names are reduced to safe relative paths; traversal, duplicate,
  binary, empty, oversized, and unsupported files fail before job creation.
- Credentials and raw database/storage URLs never appear in API responses or
  persisted manifests.
- A worker publishes a version only after the complete artifact is built,
  checksummed, and stored successfully.
