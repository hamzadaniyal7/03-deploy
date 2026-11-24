"""The command line.

Five verbs, in the order you would actually use them:

    fit      will it fit, and how much can it serve?      (no server needed)
    probe    does this server behave as advertised?       (needs a server)
    bench    what are the numbers under load?             (needs a server)
    gateway  put something sane in front of it            (long-running)
    mock     a fake backend, so the above work offline    (long-running)

`fit` needs nothing but arithmetic, which is the point: the capacity question
should be answered before anything is downloaded, started or paid for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from selfserve.bench import BenchConfig, run_bench, run_sweep
from selfserve.bench.report import BenchReport
from selfserve.capacity import plan
from selfserve.config import CONFIG_FILENAME, Config, ConfigError
from selfserve.gateway import Gateway, GatewayConfig
from selfserve.mock import MockConfig, MockServer
from selfserve.models import GPUS, MODELS, from_hf_config, get_gpu, get_model
from selfserve.probe import probe

STARTER_CONFIG = """\
# selfserve configuration -- see docs/ for what each knob does.
[selfserve]
backends = ["http://localhost:8000"]
host = "127.0.0.1"
port = 8080

# Admission control: bounded in flight, bounded waiting, fast 503 past that.
max_concurrency = 64
queue_limit = 128
queue_timeout_s = 10

# Resilience
max_retries = 2
failure_threshold = 3
recovery_s = 15

# Benchmark defaults
concurrency = 8
requests = 64
prompt_tokens = 256
max_tokens = 128

# Capacity defaults
gpu = "a100-80"
gpus = 1
dtype = "fp16"
utilization = 0.90
"""


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except (ConfigError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="selfserve",
        description="Serve an open model yourself: capacity, gateway, benchmark.",
    )
    sub = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help=f"path to {CONFIG_FILENAME}")

    # ------------------------------------------------------------------ fit
    fit = sub.add_parser("fit", parents=[common], help="capacity planning, no server needed")
    fit.add_argument("model", nargs="?", help=f"one of: {', '.join(sorted(MODELS))}")
    fit.add_argument("--hf-config", help="path to a HuggingFace config.json instead")
    fit.add_argument("--params-b", type=float, help="parameter count in billions, for --hf-config")
    fit.add_argument("--gpu", help=f"one of: {', '.join(sorted(GPUS))}")
    fit.add_argument("--gpus", type=int, help="how many of them")
    fit.add_argument("--dtype", help="fp16 | bf16 | fp8 | int8 | awq | gptq | fp32")
    fit.add_argument("--kv-dtype", help="KV cache dtype (defaults to the weight dtype)")
    fit.add_argument("--context", type=int, help="context length to plan for")
    fit.add_argument("--utilization", type=float, help="fraction of GPU memory the server may use")
    fit.add_argument("--json", action="store_true")
    fit.add_argument("--list", action="store_true", help="list known models and GPUs")
    fit.set_defaults(func=cmd_fit)

    # ---------------------------------------------------------------- probe
    pr = sub.add_parser("probe", parents=[common], help="check an endpoint for compatibility")
    pr.add_argument("url", nargs="?", help="backend URL; defaults to the configured one")
    pr.add_argument("--model", help="model id to exercise")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_probe)

    # ---------------------------------------------------------------- bench
    bench = sub.add_parser("bench", parents=[common], help="load test an endpoint")
    bench.add_argument("url", nargs="?", help="endpoint URL; defaults to the configured backend")
    bench.add_argument("--model")
    bench.add_argument("-c", "--concurrency", type=int)
    bench.add_argument("-n", "--requests", type=int)
    bench.add_argument("--prompt-tokens", type=int)
    bench.add_argument("--max-tokens", type=int)
    bench.add_argument("--sweep", help="comma-separated concurrency levels, e.g. 1,4,16,64")
    bench.add_argument("--no-stream", action="store_true")
    bench.add_argument("--warmup", type=int)
    bench.add_argument("--no-save", action="store_true")
    bench.add_argument("--json", action="store_true")
    bench.set_defaults(func=cmd_bench)

    # -------------------------------------------------------------- gateway
    gw = sub.add_parser("gateway", parents=[common], help="run the gateway (long-running)")
    gw.add_argument("--backends", help="comma-separated backend URLs")
    gw.add_argument("--host")
    gw.add_argument("--port", type=int)
    gw.add_argument("--max-concurrency", type=int)
    gw.add_argument("--queue-limit", type=int)
    gw.set_defaults(func=cmd_gateway)

    # ----------------------------------------------------------------- mock
    mock = sub.add_parser("mock", help="run a simulated backend (long-running)")
    mock.add_argument("--port", type=int, default=8000)
    mock.add_argument("--capacity", type=int, default=32, help="concurrent sequences before 503")
    mock.add_argument("--ttft", type=float, default=0.080, help="base time to first token, seconds")
    mock.add_argument("--itl", type=float, default=0.022, help="base inter-token latency, seconds")
    mock.add_argument("--failure-rate", type=float, default=0.0)
    mock.set_defaults(func=cmd_mock)

    init = sub.add_parser("init", help=f"write a starter {CONFIG_FILENAME}")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    return parser


# ------------------------------------------------------------------ commands
def cmd_fit(args: argparse.Namespace) -> int:
    if args.list:
        print("models:")
        for key, spec in sorted(MODELS.items()):
            tag = " (MoE)" if spec.is_moe else ""
            print(
                f"  {key:<16} {spec.params_b:>6.1f}B  {spec.layers:>3} layers  "
                f"{spec.kv_heads:>2} kv_heads  {spec.max_context // 1024:>4}k ctx{tag}"
            )
        print("\ngpus:")
        for key, gpu in sorted(GPUS.items(), key=lambda kv: kv[1].memory_gb):
            print(f"  {key:<16} {gpu.memory_gb:>6.0f} GB  {gpu.bandwidth_gbs:>6.0f} GB/s")
        return 0

    cfg = Config.load(
        args.config, gpu=args.gpu, gpus=args.gpus, dtype=args.dtype, utilization=args.utilization
    )

    if args.hf_config:
        raw = json.loads(Path(args.hf_config).read_text())
        model = from_hf_config(
            raw, name=args.model or Path(args.hf_config).parent.name, params_b=args.params_b
        )
    elif args.model:
        model = get_model(args.model)
    else:
        print(
            "error: give a model name, or --hf-config path/to/config.json "
            "(or --list to see what is known)",
            file=sys.stderr,
        )
        return 2

    capacity = plan(
        model,
        get_gpu(cfg.gpu),
        gpus=cfg.gpus,
        dtype=cfg.dtype,
        kv_dtype=args.kv_dtype,
        context=args.context,
        utilization=cfg.utilization,
    )
    print(json.dumps(capacity.to_dict(), indent=2) if args.json else capacity.render())
    return 0 if capacity.fits else 1


def cmd_probe(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    url = args.url or cfg.backends[0]
    report = probe(url, api_key=cfg.backend_api_key, model=args.model or cfg.model)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.render())
    return 0 if report.passed else 1


def cmd_bench(args: argparse.Namespace) -> int:
    cfg = Config.load(
        args.config,
        concurrency=args.concurrency,
        requests=args.requests,
        prompt_tokens=args.prompt_tokens,
        max_tokens=args.max_tokens,
        model=args.model,
        warmup=args.warmup,
    )
    bench_config = BenchConfig(
        url=args.url or cfg.backends[0],
        model=cfg.model,
        concurrency=cfg.concurrency,
        requests=cfg.requests,
        prompt_tokens=cfg.prompt_tokens,
        max_tokens=cfg.max_tokens,
        stream=not args.no_stream,
        api_key=cfg.backend_api_key,
        timeout=cfg.request_timeout_s,
        warmup=cfg.warmup,
    )

    if args.sweep:
        levels = [int(x) for x in args.sweep.split(",") if x.strip()]
        sweep = run_sweep(bench_config, levels, on_report=_print_level)
        print()
        print(sweep.render())
        if not args.no_save:
            _save("sweep", sweep.to_json())
        return 0

    report = run_bench(bench_config)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.render())
    if not args.no_save and not args.json:
        _save(f"c{bench_config.concurrency}", json.dumps(report.to_dict(), indent=2))
    return 0 if report.completed else 1


def cmd_gateway(args: argparse.Namespace) -> int:
    cfg = Config.load(
        args.config,
        backends=args.backends,
        host=args.host,
        port=args.port,
        max_concurrency=args.max_concurrency,
        queue_limit=args.queue_limit,
    )
    gateway = Gateway(
        GatewayConfig(
            backends=list(cfg.backends),
            host=cfg.host,
            port=cfg.port,
            api_key=cfg.api_key,
            backend_api_key=cfg.backend_api_key,
            max_concurrency=cfg.max_concurrency,
            queue_limit=cfg.queue_limit,
            queue_timeout_s=cfg.queue_timeout_s,
            request_timeout_s=cfg.request_timeout_s,
            max_retries=cfg.max_retries,
            failure_threshold=cfg.failure_threshold,
            recovery_s=cfg.recovery_s,
        )
    )
    print(f"gateway listening on http://{cfg.host}:{cfg.port}")
    for url in cfg.backends:
        print(f"  backend  {url}")
    print(f"  admission  {cfg.max_concurrency} in flight, {cfg.queue_limit} queued")
    print("  status     GET /status   metrics: GET /metrics")
    gateway.serve_forever()
    return 0


def cmd_mock(args: argparse.Namespace) -> int:
    config = MockConfig(
        base_ttft_s=args.ttft,
        base_itl_s=args.itl,
        capacity=args.capacity,
        failure_rate=args.failure_rate,
    )
    server = MockServer(config, port=args.port)
    print(
        f"mock backend on {server.url}  (capacity {config.capacity}, "
        f"ttft {config.base_ttft_s * 1000:.0f}ms, itl {config.base_itl_s * 1000:.0f}ms)"
    )
    print("  this is a simulator, not a model -- see src/selfserve/mock/server.py")
    server.start()
    try:
        import threading

        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(CONFIG_FILENAME)
    if path.exists() and not args.force:
        print(f"{path} already exists (use --force to overwrite)", file=sys.stderr)
        return 2
    path.write_text(STARTER_CONFIG, encoding="utf-8")
    print(f"wrote {path}")
    return 0


# ------------------------------------------------------------------ helpers
def _print_level(report: BenchReport) -> None:
    ttft = report.ttft()
    print(
        f"  c={report.concurrency:<4} {report.output_tokens_per_second:>8.1f} tok/s  "
        f"TTFT p50 {ttft.p50 * 1000:>6.0f}ms  p99 {ttft.p99 * 1000:>6.0f}ms  "
        f"{len(report.rejected)} rejected",
        file=sys.stderr,
        flush=True,
    )


def _save(label: str, payload: str) -> None:
    out = Path("bench/reports")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{label}.json"
    path.write_text(payload, encoding="utf-8")
    print(f"\nreport: {path}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
