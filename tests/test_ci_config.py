from __future__ import annotations

from pathlib import Path

import yaml


def _config() -> dict:
    path = Path(__file__).resolve().parents[1] / ".gitlab-ci.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_pipeline_has_blocking_quality_and_package_jobs() -> None:
    config = _config()

    assert config["stages"] == ["lint", "test", "package", "publish"]
    assert config["lint"]["script"] == ["uvx --from ruff==0.14.13 ruff check ."]
    assert "ripgrep" in config[".python-job"]["before_script"][-2]
    assert "uv run pytest -q --junitxml=reports/pytest.xml" in config["test"]["script"]
    assert config["test"]["artifacts"]["reports"]["junit"] == "reports/pytest.xml"
    assert config["package"]["artifacts"]["paths"] == ["dist/*.whl", "dist/*.tar.gz"]


def test_container_publish_is_gated_and_uses_gitlab_credentials() -> None:
    job = _config()["publish-container"]
    rules = [rule["if"] for rule in job["rules"]]
    before_script = "\n".join(job["before_script"])
    script = "\n".join(job["script"])

    assert job["tags"] == ["sh"]
    assert rules == ["$CI_COMMIT_TAG", "$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH"]
    assert "CI_REGISTRY_PASSWORD" in before_script
    assert "--password-stdin" in before_script
    assert "$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA" in script
    assert "$CI_REGISTRY_IMAGE:latest" in script
    assert "docker buildx build" in script
    assert "--push" in script
