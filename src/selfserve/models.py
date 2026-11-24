"""Model and GPU specifications.

The numbers here are architecture facts read from published model configs
(`config.json`) and vendor spec sheets. They are in a table rather than fetched
at runtime so that `selfserve fit` answers in milliseconds, offline, before you
have downloaded 16GB of weights to find out it does not fit.

**What the fields mean, and why these ones.** Serving capacity is decided by
the KV cache, and the KV cache is decided by four numbers: layers, KV heads,
head dimension, and dtype. Everything else about a model -- its MLP width, its
vocabulary, how good it is -- is irrelevant to whether it fits.

`kv_heads` is the one that catches people out. Modern models use grouped-query
attention, where many query heads share one key/value head. Llama 3.1 8B has 32
query heads and 8 KV heads, so its cache is a quarter the size you would
compute from the attention-head count. Getting this wrong overestimates memory
by 4x, which is the difference between "fits comfortably" and "OOM on the third
concurrent request".

Specs are a snapshot. `selfserve fit --config path/to/config.json` reads a real
config instead, and is what you should use for anything not in this table.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bytes per parameter / per KV element, by dtype. The quantised entries are
# weight-only: KV cache stays at fp16 unless the server is told otherwise,
# which is a distinction that matters a lot and is easy to miss.
DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "awq": 0.5,  # 4-bit weights + scales, ~0.5-0.6 in practice
    "gptq": 0.5,
    "int4": 0.5,
}


@dataclass(slots=True, frozen=True)
class ModelSpec:
    """The architecture numbers that decide how much memory serving will take."""

    name: str
    params_b: float  # billions, total
    layers: int
    hidden_size: int
    attention_heads: int
    kv_heads: int  # < attention_heads means grouped-query attention
    head_dim: int
    vocab_size: int
    max_context: int
    # MoE models load every expert into memory but only route through some of
    # them per token, so weights scale with total params while decode speed
    # scales with active params. One number cannot describe both.
    active_params_b: float | None = None

    @property
    def is_moe(self) -> bool:
        return self.active_params_b is not None

    @property
    def gqa_ratio(self) -> float:
        """Query heads per KV head. 1.0 is full multi-head attention."""
        return self.attention_heads / self.kv_heads

    def weight_bytes(self, dtype: str = "fp16") -> int:
        return int(self.params_b * 1e9 * DTYPE_BYTES[dtype])

    def kv_bytes_per_token(self, dtype: str = "fp16") -> int:
        """Bytes of KV cache one token of one sequence occupies.

            2 (K and V) x layers x kv_heads x head_dim x bytes_per_element

        The factor of 2 is keys and values, not a fudge. Everything scales
        linearly, which is why a 128k-context request is not "a bit more
        memory" than a 4k one -- it is thirty-two times more.
        """
        return 2 * self.layers * self.kv_heads * self.head_dim * int(DTYPE_BYTES[dtype])


@dataclass(slots=True, frozen=True)
class GpuSpec:
    """What you are deploying onto.

    `bandwidth_gbs` is here because decode speed is bound by memory bandwidth,
    not by compute: generating one token means reading the whole model out of
    HBM. A card with twice the FLOPS and the same bandwidth generates tokens at
    roughly the same rate.
    """

    name: str
    memory_gb: float
    bandwidth_gbs: float  # nominal HBM bandwidth

    @property
    def memory_bytes(self) -> int:
        return int(self.memory_gb * 1024**3)


# ------------------------------------------------------------------ registry
MODELS: dict[str, ModelSpec] = {
    "llama-3.2-3b": ModelSpec(
        name="Llama 3.2 3B Instruct",
        params_b=3.21,
        layers=28,
        hidden_size=3072,
        attention_heads=24,
        kv_heads=8,
        head_dim=128,
        vocab_size=128256,
        max_context=131072,
    ),
    "llama-3.1-8b": ModelSpec(
        name="Llama 3.1 8B Instruct",
        params_b=8.03,
        layers=32,
        hidden_size=4096,
        attention_heads=32,
        kv_heads=8,
        head_dim=128,
        vocab_size=128256,
        max_context=131072,
    ),
    "llama-3.1-70b": ModelSpec(
        name="Llama 3.1 70B Instruct",
        params_b=70.6,
        layers=80,
        hidden_size=8192,
        attention_heads=64,
        kv_heads=8,
        head_dim=128,
        vocab_size=128256,
        max_context=131072,
    ),
    "qwen2.5-7b": ModelSpec(
        name="Qwen2.5 7B Instruct",
        params_b=7.62,
        layers=28,
        hidden_size=3584,
        attention_heads=28,
        kv_heads=4,
        head_dim=128,
        vocab_size=152064,
        max_context=32768,
    ),
    "qwen2.5-32b": ModelSpec(
        name="Qwen2.5 32B Instruct",
        params_b=32.8,
        layers=64,
        hidden_size=5120,
        attention_heads=40,
        kv_heads=8,
        head_dim=128,
        vocab_size=152064,
        max_context=32768,
    ),
    "mistral-7b": ModelSpec(
        name="Mistral 7B Instruct v0.3",
        params_b=7.25,
        layers=32,
        hidden_size=4096,
        attention_heads=32,
        kv_heads=8,
        head_dim=128,
        vocab_size=32768,
        max_context=32768,
    ),
    "mixtral-8x7b": ModelSpec(
        name="Mixtral 8x7B Instruct",
        params_b=46.7,
        layers=32,
        hidden_size=4096,
        attention_heads=32,
        kv_heads=8,
        head_dim=128,
        vocab_size=32000,
        max_context=32768,
        active_params_b=12.9,
    ),
    "gemma-2-9b": ModelSpec(
        name="Gemma 2 9B Instruct",
        params_b=9.24,
        layers=42,
        hidden_size=3584,
        attention_heads=16,
        kv_heads=8,
        head_dim=256,
        vocab_size=256000,
        max_context=8192,
    ),
}

GPUS: dict[str, GpuSpec] = {
    "l4": GpuSpec("NVIDIA L4", 24, 300),
    "a10g": GpuSpec("NVIDIA A10G", 24, 600),
    "rtx-4090": GpuSpec("NVIDIA RTX 4090", 24, 1008),
    "a100-40": GpuSpec("NVIDIA A100 40GB", 40, 1555),
    "l40s": GpuSpec("NVIDIA L40S", 48, 864),
    "a100-80": GpuSpec("NVIDIA A100 80GB", 80, 2039),
    "h100": GpuSpec("NVIDIA H100 80GB SXM", 80, 3350),
    "h200": GpuSpec("NVIDIA H200 141GB", 141, 4800),
}


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key.lower()]
    except KeyError:
        raise KeyError(
            f"unknown model {key!r}. Known: {', '.join(sorted(MODELS))}. "
            "For anything else use --config path/to/config.json"
        ) from None


def get_gpu(key: str) -> GpuSpec:
    try:
        return GPUS[key.lower()]
    except KeyError:
        raise KeyError(f"unknown gpu {key!r}. Known: {', '.join(sorted(GPUS))}") from None


def from_hf_config(config: dict, name: str = "custom", params_b: float | None = None) -> ModelSpec:
    """Build a ModelSpec from a HuggingFace `config.json`.

    Field names vary across model families, so each lookup takes the first key
    that is present. `head_dim` is often absent and has to be derived from
    hidden_size / attention heads -- Gemma 2 is the counter-example that makes
    the explicit field worth checking first, since its head_dim is 256 while
    hidden_size / heads would give 224.
    """
    layers = _first(config, "num_hidden_layers", "n_layer", "num_layers")
    hidden = _first(config, "hidden_size", "n_embd", "d_model")
    heads = _first(config, "num_attention_heads", "n_head")
    kv_heads = _first(config, "num_key_value_heads", "num_kv_heads", default=heads)
    head_dim = _first(config, "head_dim", default=hidden // heads)

    if params_b is None:
        params_b = _estimate_params_b(layers, hidden, config)

    return ModelSpec(
        name=name,
        params_b=params_b,
        layers=layers,
        hidden_size=hidden,
        attention_heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=_first(config, "vocab_size", default=32000),
        max_context=_first(config, "max_position_embeddings", "max_seq_len", default=4096),
    )


def _first(config: dict, *keys: str, default: int | None = None) -> int:
    for key in keys:
        if config.get(key) is not None:
            return int(config[key])
    if default is None:
        raise KeyError(f"config is missing all of: {', '.join(keys)}")
    return int(default)


def _estimate_params_b(layers: int, hidden: int, config: dict) -> float:
    """Rough parameter count from the architecture.

    Transformer blocks plus embeddings, ignoring biases and norms. Within a few
    percent for dense models, and wrong for MoE -- pass `params_b` explicitly
    there rather than trusting this.
    """
    intermediate = int(config.get("intermediate_size") or 4 * hidden)
    vocab = int(config.get("vocab_size") or 32000)
    attn = 4 * hidden * hidden
    mlp = 3 * hidden * intermediate  # gate, up, down (SwiGLU)
    return (layers * (attn + mlp) + 2 * vocab * hidden) / 1e9
