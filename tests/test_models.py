"""The model registry and reading real configs."""

from __future__ import annotations

import pytest

from selfserve.models import DTYPE_BYTES, GPUS, MODELS, from_hf_config, get_gpu, get_model


def test_every_registry_entry_is_internally_consistent() -> None:
    for key, spec in MODELS.items():
        assert spec.kv_heads <= spec.attention_heads, key
        assert spec.attention_heads % spec.kv_heads == 0, f"{key}: GQA groups must divide evenly"
        assert spec.layers > 0 and spec.head_dim > 0, key
        assert spec.max_context >= 2048, key
        assert spec.params_b > 0, key


def test_gemma_head_dim_is_not_hidden_over_heads() -> None:
    """The counter-example that makes the explicit head_dim field necessary."""
    spec = get_model("gemma-2-9b")
    assert spec.head_dim == 256
    assert spec.hidden_size // spec.attention_heads == 224


def test_moe_declares_active_parameters() -> None:
    spec = get_model("mixtral-8x7b")
    assert spec.is_moe and spec.active_params_b < spec.params_b


def test_dense_models_are_not_moe() -> None:
    assert not get_model("llama-3.1-8b").is_moe


def test_unknown_model_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="llama-3.1-8b"):
        get_model("gpt-5")


def test_unknown_gpu_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="h100"):
        get_gpu("tpu-v5")


def test_lookup_is_case_insensitive() -> None:
    assert get_model("LLAMA-3.1-8B") is get_model("llama-3.1-8b")
    assert get_gpu("H100") is get_gpu("h100")


def test_every_gpu_has_plausible_specs() -> None:
    for key, gpu in GPUS.items():
        assert 8 <= gpu.memory_gb <= 200, key
        assert 100 <= gpu.bandwidth_gbs <= 10_000, key
        assert gpu.memory_bytes == int(gpu.memory_gb * 1024**3)


def test_quantised_dtypes_are_smaller_than_fp16() -> None:
    assert DTYPE_BYTES["awq"] < DTYPE_BYTES["int8"] < DTYPE_BYTES["fp16"] < DTYPE_BYTES["fp32"]


# ---------------------------------------------------------------- hf config
LLAMA_CONFIG = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
}


def test_reads_a_real_config() -> None:
    spec = from_hf_config(LLAMA_CONFIG, name="llama-ish", params_b=8.03)
    reference = get_model("llama-3.1-8b")
    assert spec.kv_bytes_per_token("fp16") == reference.kv_bytes_per_token("fp16")


def test_derives_head_dim_when_absent() -> None:
    spec = from_hf_config(LLAMA_CONFIG)
    assert spec.head_dim == 4096 // 32


def test_prefers_an_explicit_head_dim() -> None:
    spec = from_hf_config({**LLAMA_CONFIG, "head_dim": 256})
    assert spec.head_dim == 256


def test_defaults_kv_heads_to_attention_heads_for_pre_gqa_models() -> None:
    config = {k: v for k, v in LLAMA_CONFIG.items() if k != "num_key_value_heads"}
    assert from_hf_config(config).kv_heads == 32


def test_accepts_alternative_field_names() -> None:
    spec = from_hf_config(
        {"n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257, "max_seq_len": 1024}
    )
    assert spec.layers == 12 and spec.hidden_size == 768


def test_parameter_estimate_is_in_the_right_region() -> None:
    """Within a few percent for dense models is all this needs to be."""
    spec = from_hf_config(LLAMA_CONFIG)
    assert 7.0 < spec.params_b < 9.0


def test_a_config_missing_essentials_fails_loudly() -> None:
    with pytest.raises(KeyError, match="missing"):
        from_hf_config({"hidden_size": 4096})
