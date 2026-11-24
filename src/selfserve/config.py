"""Configuration: one dataclass, three sources, documented precedence.

Precedence (last wins):
  1. the defaults below
  2. ``selfserve.toml`` in the current directory (or the path given to ``load``)
  3. ``SELFSERVE_*`` environment variables
  4. explicit keyword overrides (what the CLI flags become)

Environment variables matter more here than in the other two projects in this
series: a gateway runs in a container, and a container is configured by its
environment. Everything the gateway needs is settable without a config file.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "selfserve.toml"


class ConfigError(ValueError):
    """Invalid configuration, raised before anything starts listening."""


@dataclass(slots=True)
class Config:
    # --- gateway -----------------------------------------------------------
    backends: tuple[str, ...] = ("http://localhost:8000",)
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str | None = None  # required from clients, if set
    backend_api_key: str | None = None  # presented to backends, if set

    # --- admission ---------------------------------------------------------
    max_concurrency: int = 64
    queue_limit: int = 128
    queue_timeout_s: float = 10.0
    request_timeout_s: float = 300.0

    # --- resilience --------------------------------------------------------
    max_retries: int = 2
    failure_threshold: int = 3
    recovery_s: float = 15.0

    # --- benchmark ---------------------------------------------------------
    model: str = "default"
    concurrency: int = 8
    requests: int = 64
    prompt_tokens: int = 256
    max_tokens: int = 128
    warmup: int = 2

    # --- capacity ----------------------------------------------------------
    gpu: str = "a100-80"
    gpus: int = 1
    dtype: str = "fp16"
    utilization: float = 0.90

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> Config:
        cfg = cls()
        toml_path = Path(path) if path else Path(CONFIG_FILENAME)
        if toml_path.is_file():
            with toml_path.open("rb") as fh:
                data = tomllib.load(fh)
            cfg = cfg.merged(data.get("selfserve", data))
        cfg = cfg.merged(cls._from_env())
        if overrides:
            cfg = cfg.merged({k: v for k, v in overrides.items() if v is not None})
        cfg.validate()
        return cfg

    @staticmethod
    def _from_env() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(Config):
            raw = os.environ.get(f"SELFSERVE_{f.name.upper()}")
            if raw is not None:
                out[f.name] = raw
        return out

    def merged(self, data: dict[str, Any]) -> Config:
        known = {f.name for f in fields(Config)}
        current = {f.name: getattr(self, f.name) for f in fields(Config)}
        for key, value in data.items():
            if key not in known:
                raise ConfigError(f"unknown config key: {key!r}")
            current[key] = _coerce(value, current[key], key)
        return Config(**current)

    def validate(self) -> None:
        if not self.backends:
            raise ConfigError("at least one backend is required")
        for url in self.backends:
            if not url.startswith(("http://", "https://")):
                raise ConfigError(f"backend {url!r} must start with http:// or https://")
        if self.max_concurrency < 1:
            raise ConfigError("max_concurrency must be >= 1")
        if self.queue_limit < 0:
            raise ConfigError("queue_limit must be >= 0")
        if self.concurrency < 1:
            raise ConfigError("concurrency must be >= 1")
        if self.requests < 1:
            raise ConfigError("requests must be >= 1")
        if not 0 < self.utilization <= 1:
            raise ConfigError("utilization must be in (0, 1]")
        if self.gpus < 1:
            raise ConfigError("gpus must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(Config)}


def _coerce(value: Any, like: Any, key: str) -> Any:
    """Coerce a TOML/env value to the type of the existing default."""
    if key in {"api_key", "backend_api_key"}:
        return str(value) or None if value else None
    if isinstance(like, bool):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(like, int) and not isinstance(like, bool):
        return int(value)
    if isinstance(like, float):
        return float(value)
    if isinstance(like, tuple):
        if isinstance(value, str):
            return tuple(p.strip() for p in value.split(",") if p.strip())
        return tuple(value)
    return value
