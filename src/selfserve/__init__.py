"""selfserve -- serve an open model yourself, and know the numbers.

from selfserve import plan, get_model, get_gpu
print(plan(get_model("llama-3.1-8b"), get_gpu("a100-80")).render())
"""

from __future__ import annotations

from selfserve.bench import BenchConfig, BenchReport, Sweep, run_bench, run_sweep
from selfserve.capacity import Capacity, plan
from selfserve.client import ChatRequest, InferenceClient
from selfserve.config import Config, ConfigError
from selfserve.gateway import AdmissionControl, Backend, BackendPool, Gateway, GatewayConfig, Router
from selfserve.mock import MockConfig, MockServer
from selfserve.models import GPUS, MODELS, GpuSpec, ModelSpec, from_hf_config, get_gpu, get_model
from selfserve.probe import ProbeReport, probe
from selfserve.types import Percentiles, RequestResult

__version__ = "0.1.0"

__all__ = [
    "GPUS",
    "MODELS",
    "AdmissionControl",
    "Backend",
    "BackendPool",
    "BenchConfig",
    "BenchReport",
    "Capacity",
    "ChatRequest",
    "Config",
    "ConfigError",
    "Gateway",
    "GatewayConfig",
    "GpuSpec",
    "InferenceClient",
    "MockConfig",
    "MockServer",
    "ModelSpec",
    "Percentiles",
    "ProbeReport",
    "RequestResult",
    "Router",
    "Sweep",
    "__version__",
    "from_hf_config",
    "get_gpu",
    "get_model",
    "plan",
    "probe",
    "run_bench",
    "run_sweep",
]
