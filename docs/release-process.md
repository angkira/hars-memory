# Release process

This package is distributed through two channels:

1. **GitLab PyPI registry** — Python package (`hars-longterm-memory` wheel + sdist)
2. **GitLab Container Registry** — Docker image for the HTTP index service

Both use the GitLab project at `code.knotgate.com/hars/hars-longterm-memory`
(project id `14`).

---

## Versioning

The package uses [semantic versioning](https://semver.org/). Current version is
in `pyproject.toml` (`version = "0.5.0"`).

---

## Publishing a new version

### GitHub-tag release (current process)

There is no PyPI index — distribution is GitHub-only by decision. Pushing a
version tag (e.g. `v0.9.0`) starts a CI pipeline (lint → test → package, plus
the `publish-container` image job); consumers install straight from the public
repo (see "Consumers" below, and `docs/integration-guide.md`).

Release checklist (local part, before tagging):

1. **Check current version**: confirm no one else already bumped it
   (`git tag --list | sort -V | tail -3`, and `CHANGELOG.md` top entry).

2. **Bump version** in `pyproject.toml`:
   ```bash
   # Edit pyproject.toml, change version
   uv lock  # updates uv.lock
   ```

3. **Run tests**: must pass:
   ```bash
   uv sync --extra dev
   uv run pytest -q
   ```

4. **Tag the commit**:
   ```bash
   git add pyproject.toml uv.lock
   git commit -m "Bump version to X.Y.Z"
   git tag vX.Y.Z
   ```

5. **Build**:
   ```bash
   uv build
   # Produces: dist/hars_longterm_memory-X.Y.Z-py3-none-any.whl
   #            dist/hars_longterm_memory-X.Y.Z.tar.gz
   ```

6. **Publish**: there is no registry upload — pushing the tag (step 8) IS the
   release. CI validates it (lint → test → package) and builds the container
   image. (Historical note: a retired `publish-pypi` job once uploaded to a
   GitLab PyPI registry; do not resurrect it — distribution is GitHub-only
   by decision.)

7. **Verify, do not trust**: after CI goes green, confirm the tag and the
   package artifacts exist:
   ```bash
   git ls-remote --tags origin | grep 'vX.Y.Z'
   # + check the tag pipeline (lint/test/package green) in CI.
   ```

8. **Push the tag** — this starts the CI release pipeline (lint → test →
   package + container image):
   ```bash
   git push origin vX.Y.Z
   ```

### CI/CD (automatic)

The `.gitlab-ci.yml` pipeline runs on every branch, MR, and tag:

| Stage | What it does | Runs on |
|---|---|---|
| `lint` | Ruff check | All branches, MRs, tags |
| `test` | Full pytest suite (live LLM tests opt-in/skipped) | All branches, MRs, tags |
| `package` | Builds wheel + sdist | All branches, MRs, tags |
| `publish-container` | Builds `Dockerfile.service` and pushes to GitLab Container Registry | Tags + default branch |

**Container image tags:**

| Tag | Description |
|---|---|
| `$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA` | Immutable provenance tag |
| `$CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA` | Operator-friendly commit tag |
| `$CI_REGISTRY_IMAGE:$CI_COMMIT_REF_SLUG` | Tag pipelines only |
| `$CI_REGISTRY_IMAGE:latest` | Default branch only |

There is no PyPI publishing. Consumers install from GitHub (section below).

---

## Consumers

### How consumers install from GitHub (no registry)

```toml
[project]
dependencies = [
  "hars-longterm-memory @ git+https://github.com/angkira/hars-memory@vX.Y.Z",
  "torch>=2.0.0",  # explicit CPU-only override required, see integration guide
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

See `docs/integration-guide.md` for the full consumer setup.

### Installing the container image

```bash
docker login registry.gitlab.com -u <deploy-token-username> -p <deploy-token-password>
docker pull registry.gitlab.com/hars/hars-longterm-memory:latest
```

---

## Distribution channels overview

| Channel | Artifact | Published by | Consumer access |
|---|---|---|---|
| GitHub repo (tags) | Source + release tag | `git push origin vX.Y.Z` | Public, anonymous |
| GitLab Container Registry | Docker image (`Dockerfile.service`) | CI (tags + default branch) | Deploy token or anonymous |
| Git repo | Source code | `git push` | SSH key / public HTTPS |
| `.claude/skills/` | Portable skill for AI agents | Manual copy | File copy |

## Credential model

No registry credentials are needed: the repo is public and consumers install
via plain HTTPS git URLs. The only secrets in play are deploy-time ones
(LLM API keys, access tokens for a deployment's own MCP server) — those live
in gitignored secrets files on the operator's machine, never in this repo,
never in a commit message, never in CI config checked into git. CI needs no
stored tokens: the container job uses auto-injected short-lived
`CI_REGISTRY_*` credentials.