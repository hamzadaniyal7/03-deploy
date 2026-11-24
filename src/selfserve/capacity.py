"""Will it fit, and how much can it serve?

This is arithmetic, not a benchmark, and it is the arithmetic that decides
whether a deployment works. GPU memory goes to three places:

    weights  +  KV cache  +  everything else

`everything else` is activations, CUDA context, fragmentation and the
allocator's own overhead. vLLM's `--gpu-memory-utilization` (default 0.9) is a
cap on the *total* the process may take, and whatever is left after weights
becomes the KV cache. So the useful question is never "does the model fit" --
an 8B model in fp16 is 16GB and obviously fits on an 80GB card. The question is
**how much KV cache is left afterwards**, because that is what sets how many
requests you can serve at once.

A worked example, which is the thing to take away:

    Llama 3.1 8B, fp16, on an A100 80GB at 0.90 utilisation
      total usable   72.0 GB
      weights        15.0 GB
      overhead        2.0 GB
      KV cache       55.0 GB
      per token     128 KB   (2 x 32 layers x 8 kv_heads x 128 dim x 2 bytes)
      ------------------------------------------------------------------
      at   4k context:  110 concurrent sequences
      at  32k context:   13
      at 128k context:    3

The model did not change. The context length did, and it moved the answer by
nearly 40x. Anyone sizing a deployment from "how big is the model" is going to
be surprised in production, and this module exists so that the surprise happens
at the command line instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from selfserve.models import DTYPE_BYTES, GpuSpec, ModelSpec

GB = 1024**3

# Non-weight, non-KV reserve: CUDA context, activation buffers, allocator
# fragmentation. Scales with batch and hidden size, but a flat couple of GB is
# close enough for sizing and errs the safe way.
DEFAULT_OVERHEAD_GB = 2.0


@dataclass(slots=True)
class Capacity:
    """The answer, with every intermediate value kept.

    Intermediates are kept deliberately: a capacity number you cannot
    decompose is a number nobody will act on, because the first question is
    always "where did the memory go?".
    """

    model: ModelSpec
    gpu: GpuSpec
    gpus: int
    dtype: str
    kv_dtype: str
    context: int
    utilization: float

    total_bytes: int = 0
    usable_bytes: int = 0
    weight_bytes: int = 0
    overhead_bytes: int = 0
    kv_bytes: int = 0
    kv_per_token: int = 0

    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- results
    @property
    def fits(self) -> bool:
        return self.kv_bytes > 0

    @property
    def max_sequences(self) -> int:
        """Concurrent sequences at the full context length.

        The pessimistic number: it assumes every sequence runs to the context
        limit. Real traffic rarely does, which is what `max_tokens` is for.
        """
        if not self.fits or self.context <= 0:
            return 0
        return int(self.kv_bytes // (self.kv_per_token * self.context))

    @property
    def max_tokens(self) -> int:
        """Total KV tokens in flight, across all sequences.

        This is the number vLLM actually works in, and the honest one to plan
        with: twenty sequences averaging 2k tokens cost the same as one at 40k.
        """
        if not self.fits:
            return 0
        return int(self.kv_bytes // self.kv_per_token)

    def sequences_at(self, context: int) -> int:
        if not self.fits or context <= 0:
            return 0
        return int(self.kv_bytes // (self.kv_per_token * context))

    def decode_tokens_per_second(self) -> float:
        """Single-stream decode speed, estimated from memory bandwidth.

        Generating one token requires reading every weight the forward pass
        touches out of HBM, so the ceiling is bandwidth / bytes-read. For MoE,
        only the routed experts are read, so active parameters are what count.

        This is an upper bound and real servers land well under it -- it
        ignores attention over the KV cache, kernel launch overhead and
        everything else. It is useful for the order of magnitude and for the
        comparison it makes obvious: batching is what turns a bandwidth-bound
        decode into a compute-bound one, which is why continuous batching is
        the single biggest throughput lever in a serving stack.
        """
        active = self.model.active_params_b or self.model.params_b
        bytes_per_token = active * 1e9 * DTYPE_BYTES[self.dtype]
        total_bandwidth = self.gpu.bandwidth_gbs * 1e9 * self.gpus
        return total_bandwidth / bytes_per_token

    # ------------------------------------------------------------ rendering
    def render(self) -> str:
        lines = [
            f"{self.model.name} @ {self.dtype}"
            + (f" (KV {self.kv_dtype})" if self.kv_dtype != self.dtype else ""),
            f"{self.gpus}x {self.gpu.name}  --  utilisation {self.utilization:.2f}",
            "",
            f"  total memory      {_gb(self.total_bytes):>8}",
            f"  usable            {_gb(self.usable_bytes):>8}   (x{self.utilization:.2f})",
            f"  weights           {_gb(self.weight_bytes):>8}",
            f"  overhead          {_gb(self.overhead_bytes):>8}",
            f"  KV cache          {_gb(self.kv_bytes):>8}"
            + ("" if self.fits else "   << negative"),
            "",
        ]
        if not self.fits:
            lines.append("  DOES NOT FIT: weights plus overhead exceed usable memory.")
            lines.append("  Try: more GPUs (--gpus), a quantised dtype (--dtype awq),")
            lines.append("  or a smaller model.")
            return "\n".join(lines + [""] + [f"  ! {w}" for w in self.warnings])

        lines += [
            f"  KV per token      {_kb(self.kv_per_token):>8}"
            f"   (2 x {self.model.layers} layers x {self.model.kv_heads} kv_heads"
            f" x {self.model.head_dim} dim x {int(DTYPE_BYTES[self.kv_dtype])}B)",
            f"  KV tokens total   {self.max_tokens:>8,}",
            "",
            "  concurrent sequences by context length:",
        ]
        for context in (2048, 4096, 8192, 16384, 32768, 131072):
            if context > self.model.max_context:
                continue
            count = self.sequences_at(context)
            marker = "  <- configured" if context == self.context else ""
            lines.append(f"    {_ctx(context):>6}  {count:>6}{marker}")

        lines += [
            "",
            f"  decode ceiling    {self.decode_tokens_per_second():>8.0f} tok/s"
            "   single stream, memory-bound estimate",
        ]
        if self.warnings:
            lines += [""] + [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model": self.model.name,
            "gpu": self.gpu.name,
            "gpus": self.gpus,
            "dtype": self.dtype,
            "kv_dtype": self.kv_dtype,
            "context": self.context,
            "utilization": self.utilization,
            "fits": self.fits,
            "bytes": {
                "total": self.total_bytes,
                "usable": self.usable_bytes,
                "weights": self.weight_bytes,
                "overhead": self.overhead_bytes,
                "kv_cache": self.kv_bytes,
                "kv_per_token": self.kv_per_token,
            },
            "max_sequences": self.max_sequences,
            "max_tokens": self.max_tokens,
            "decode_tokens_per_second": round(self.decode_tokens_per_second(), 1),
            "warnings": self.warnings,
        }


def plan(
    model: ModelSpec,
    gpu: GpuSpec,
    *,
    gpus: int = 1,
    dtype: str = "fp16",
    kv_dtype: str | None = None,
    context: int | None = None,
    utilization: float = 0.90,
    overhead_gb: float = DEFAULT_OVERHEAD_GB,
) -> Capacity:
    """Work out what is left for the KV cache, and what that buys."""
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}; known: {', '.join(sorted(DTYPE_BYTES))}")
    kv_dtype = kv_dtype or ("fp16" if dtype in {"awq", "gptq", "int4", "int8"} else dtype)
    if kv_dtype not in DTYPE_BYTES:
        raise ValueError(f"unknown kv_dtype {kv_dtype!r}")
    if gpus < 1:
        raise ValueError("gpus must be >= 1")
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be in (0, 1]")

    context = context or model.max_context
    cap = Capacity(
        model=model,
        gpu=gpu,
        gpus=gpus,
        dtype=dtype,
        kv_dtype=kv_dtype,
        context=context,
        utilization=utilization,
    )

    cap.total_bytes = gpu.memory_bytes * gpus
    cap.usable_bytes = int(cap.total_bytes * utilization)
    cap.weight_bytes = model.weight_bytes(dtype)
    # Overhead is per GPU: each device has its own CUDA context and buffers.
    cap.overhead_bytes = int(overhead_gb * GB * gpus)
    cap.kv_bytes = cap.usable_bytes - cap.weight_bytes - cap.overhead_bytes
    cap.kv_per_token = model.kv_bytes_per_token(kv_dtype)

    cap.warnings = _warnings(cap, model, context)
    return cap


def _warnings(cap: Capacity, model: ModelSpec, context: int) -> list[str]:
    """Things that are arithmetically fine and operationally a bad idea."""
    out: list[str] = []

    if context > model.max_context:
        out.append(
            f"requested context {context:,} exceeds the model's {model.max_context:,}; "
            "the server will refuse to start or silently clamp it"
        )
    if cap.fits and cap.max_sequences == 0:
        out.append(
            f"not one sequence fits at {context:,} tokens -- a single request will OOM. "
            "Lower --max-model-len or add GPUs"
        )
    elif cap.fits and 0 < cap.max_sequences < 4:
        out.append(
            f"only {cap.max_sequences} concurrent sequence(s) at {context:,} tokens; "
            "continuous batching has nothing to batch and throughput will be poor"
        )
    if cap.gpus > 1 and model.layers % cap.gpus:
        out.append(
            f"{model.layers} layers do not divide evenly across {cap.gpus} GPUs; "
            "pipeline parallelism will be unbalanced (tensor parallelism is fine)"
        )
    if cap.gpus > 1 and model.kv_heads % cap.gpus:
        out.append(
            f"{model.kv_heads} KV heads do not divide across {cap.gpus} GPUs; "
            "tensor parallelism requires kv_heads to be divisible by the TP size"
        )
    if cap.utilization > 0.95:
        out.append(
            f"utilisation {cap.utilization:.2f} leaves almost nothing for fragmentation; "
            "0.90 is the usual safe ceiling"
        )
    if model.is_moe:
        out.append(
            f"MoE: all {model.params_b:.1f}B parameters occupy memory while only "
            f"{model.active_params_b:.1f}B are read per token, so memory is sized by the "
            "former and speed by the latter"
        )
    return out


# ----------------------------------------------------------------- display
def _gb(value: int) -> str:
    return f"{value / GB:.1f} GB"


def _kb(value: int) -> str:
    return f"{value / 1024:.0f} KB" if value < 1024**2 else f"{value / 1024**2:.1f} MB"


def _ctx(tokens: int) -> str:
    return f"{tokens // 1024}k" if tokens >= 1024 else str(tokens)
