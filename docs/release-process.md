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

### Manual PyPI publish (current process)

There is no CI job for PyPI publishing yet — it is a manual, local operation:

1. **Check current version**: confirm no one else already bumped it:
   ```bash
   curl -s "https://code.knotgate.com/api/v4/projects/14/packages" \
       -H "PRIVATE-TOKEN: $GITLAB_TOKEN" \
       | python3 -c "import json,sys; [print(p['name'], p['version']) for p in json.load(sys.stdin)]"
   ```

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

6. **Publish to GitLab PyPI registry**:
   ```bash
   uv publish \
       --publish-url https://code.knotgate.com/api/v4/projects/14/packages/pypi \
       --username __token__ \
       --password "$GITLAB_TOKEN_WITH_write_package_registry_SCOPE"
   ```

   The token needs the `write_package_registry` scope. Store it in a gitignored
   secrets file — never commit it.

7. **Verify, do not trust**: re-query the registry:
   ```bash
   curl -s "https://code.knotgate.com/api/v4/projects/14/packages" \
       -H "PRIVATE-TOKEN: $GITLAB_TOKEN" \
       | python3 -c "import json,sys; [print(p['name'], p['version']) for p in json.load(sys.stdin)]"
   ```

8. **Push the tag**:
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

**PyPI publishing is not yet automated in CI.** See "Future: automated PyPI publishing" below.

---

## Future: automated PyPI publishing in CI

A `publish-pypi` job can be added to `.gitlab-ci.yml` that runs on tags:

```yaml
publish-pypi:
  stage: publish
  needs:
    - job: lint
    - job: test
    - job: package
      artifacts: true
  rules:
    - if: '$CI_COMMIT_TAG'
  script:
    - uv publish --publish-url "$CI_PYPI_PUBLISH_URL" \
        --username "__token__" \
        --password "$CI_PYPI_PUBLISH_TOKEN"
```

The `CI_PYPI_PUBLISH_TOKEN` would need to be a GitLab CI/CD variable with the
`write_package_registry` scope. This is not yet implemented — the current
manual process requires a personal token.

---

## Consumers

### How consumers install from the registry

Consumers add the following to their `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "hars-longterm-memory"
url = "https://code.knotgate.com/api/v4/projects/14/packages/pypi/simple"
explicit = true

[project]
dependencies = ["hars-longterm-memory>=0.5.0"]

[tool.uv.sources]
hars-longterm-memory = { index = "hars-longterm-memory" }
```

Authentication is via a deploy token (not a personal token):

```bash
export UV_INDEX_HARS_LONGTERM_MEMORY_USERNAME="<deploy-token-username>"
export UV_INDEX_HARS_LONGTERM_MEMORY_PASSWORD="<deploy-token-password>"
```

### How to create a deploy token

1. Go to `code.knotgate.com/hars/hars-longterm-memory/-/settings/access_tokens`
2. Create a token with scope `read_package_registry`
3. Share the username and password securely with the consumer

### Installing the container image

```bash
docker login registry.gitlab.com -u <deploy-token-username> -p <deploy-token-password>
docker pull registry.gitlab.com/hars/hars-longterm-memory:latest
```

---

## Distribution channels overview

| Channel | Artifact | Published by | Consumer access |
|---|---|---|---|
| GitLab PyPI registry | `hars-longterm-memory` wheel + sdist | Manual (`uv publish`) | Deploy token |
| GitLab Container Registry | Docker image (`Dockerfile.service`) | CI (tags + default branch) | Deploy token or anonymous |
| Git repo | Source code | `git push` | SSH key |
| `.claude/skills/` | Portable skill for AI agents | Manual copy | File copy |

## Credential model

| Role | Credential type | Scope | Where stored |
|---|---|---|---|
| Publisher | Personal/Project access token | `write_package_registry` | Gitignored secrets file (e.g. `.env.secrets`) |
| Consumer | Deploy token | `read_package_registry` | Consumer's own secrets file |
| CI | GitLab CI_REGISTRY_* (short-lived) | GitLab Container Registry | CI/CD variables (auto-injected) |