"""Compatibility probing, and configuration precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from selfserve.config import Config, ConfigError
from selfserve.mock import MockConfig, MockServer
from selfserve.probe import probe


# ------------------------------------------------------------------- probe
def test_a_healthy_server_passes(mock: MockServer) -> None:
    report = probe(mock.url)
    assert report.passed
    assert "usable" in report.render()


def test_every_check_runs_against_a_good_server(mock: MockServer) -> None:
    names = {c.name for c in probe(mock.url).checks}
    assert {"health endpoint", "GET /v1/models", "streaming completion"} <= names


def test_usage_reporting_is_detected(mock: MockServer) -> None:
    check = next(c for c in probe(mock.url).checks if c.name == "usage in stream")
    assert check.ok and "from the server" in check.detail


def test_incremental_delivery_is_detected(mock: MockServer) -> None:
    """Catches a reverse proxy buffering the stream -- the endpoint works and
    streaming does not, which is invisible until a user complains."""
    check = next(c for c in probe(mock.url).checks if "incrementally" in c.name)
    assert check.ok


def test_an_unreachable_server_fails_fast() -> None:
    report = probe("http://127.0.0.1:9")
    assert not report.passed
    assert len(report.checks) == 1  # nothing else is meaningful
    assert "NOT usable" in report.render()


def test_a_broken_server_fails_the_completion_checks() -> None:
    with MockServer(MockConfig(failure_rate=1.0)) as server:
        report = probe(server.url)
    assert not report.passed
    failed = [c.name for c in report.checks if not c.ok and not c.advisory]
    assert "non-streaming completion" in failed


def test_probe_serialises(mock: MockServer) -> None:
    data = probe(mock.url).to_dict()
    assert data["passed"] is True
    assert all({"name", "ok", "detail", "advisory"} == set(c) for c in data["checks"])


# ------------------------------------------------------------------ config
def test_defaults_validate() -> None:
    Config().validate()


def test_toml_overrides_defaults(tmp_path: Path) -> None:
    path = tmp_path / "selfserve.toml"
    path.write_text("[selfserve]\nport = 9000\nmax_concurrency = 8\n")
    cfg = Config.load(path)
    assert cfg.port == 9000 and cfg.max_concurrency == 8


def test_env_overrides_toml(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "selfserve.toml"
    path.write_text("[selfserve]\nport = 9000\n")
    monkeypatch.setenv("SELFSERVE_PORT", "7000")
    assert Config.load(path).port == 7000


def test_explicit_overrides_win(monkeypatch) -> None:
    monkeypatch.setenv("SELFSERVE_PORT", "7000")
    assert Config.load(None, port=1234).port == 1234


def test_none_overrides_are_ignored() -> None:
    """CLI flags arrive as None when unset and must not clobber the config."""
    assert Config.load(None, port=None).port == Config().port


def test_backends_accept_a_comma_separated_string(monkeypatch) -> None:
    """A container is configured by its environment, so this has to work."""
    monkeypatch.setenv("SELFSERVE_BACKENDS", "http://a:8000, http://b:8000")
    assert Config.load(None).backends == ("http://a:8000", "http://b:8000")


def test_an_empty_api_key_env_var_means_unset(monkeypatch) -> None:
    monkeypatch.setenv("SELFSERVE_API_KEY", "")
    assert Config.load(None).api_key is None


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown config key"):
        Config().merged({"nonsense": 1})


@pytest.mark.parametrize(
    "bad",
    [
        {"backends": ()},
        {"backends": ("ftp://nope",)},
        {"max_concurrency": 0},
        {"concurrency": 0},
        {"requests": 0},
        {"utilization": 0},
        {"gpus": 0},
    ],
)
def test_invalid_values_fail_before_anything_starts(bad: dict) -> None:
    with pytest.raises(ConfigError):
        Config().merged(bad).validate()


def test_a_backend_without_a_scheme_is_rejected() -> None:
    """urllib would treat it as a relative path and fail far from the cause."""
    with pytest.raises(ConfigError, match="http://"):
        Config().merged({"backends": ("localhost:8000",)}).validate()
