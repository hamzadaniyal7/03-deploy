"""The arithmetic that decides whether a deployment works."""

from __future__ import annotations

import pytest

from selfserve.capacity import GB, plan
from selfserve.models import get_gpu, get_model


def test_kv_per_token_is_the_documented_formula() -> None:
    """2 (K and V) x layers x kv_heads x head_dim x bytes."""
    model = get_model("llama-3.1-8b")
    assert model.kv_bytes_per_token("fp16") == 2 * 32 * 8 * 128 * 2
    assert model.kv_bytes_per_token("fp16") == 131_072  # 128 KB


def test_grouped_query_attention_is_not_ignored() -> None:
    """Sizing from attention heads instead of KV heads overestimates Llama
    3.1 8B's cache by 4x -- the difference between fitting and OOM."""
    model = get_model("llama-3.1-8b")
    assert model.kv_heads == 8 and model.attention_heads == 32
    assert model.gqa_ratio == 4.0
    naive = 2 * model.layers * model.attention_heads * model.head_dim * 2
    assert naive == 4 * model.kv_bytes_per_token("fp16")


def test_fp8_kv_cache_halves_the_per_token_cost() -> None:
    model = get_model("llama-3.1-8b")
    assert model.kv_bytes_per_token("fp8") * 2 == model.kv_bytes_per_token("fp16")


def test_the_worked_example_from_the_docstring() -> None:
    """Llama 3.1 8B, fp16, A100 80GB, 0.90 -- the numbers the module documents."""
    cap = plan(get_model("llama-3.1-8b"), get_gpu("a100-80"), dtype="fp16", context=4096)
    assert round(cap.usable_bytes / GB) == 72
    assert round(cap.weight_bytes / GB) == 15
    assert round(cap.kv_bytes / GB) == 55
    assert cap.sequences_at(4096) == 110
    assert cap.sequences_at(32768) == 13
    assert cap.sequences_at(131072) == 3


def test_context_length_dominates_the_answer() -> None:
    """The point of the module: same model, same card, 32x the difference."""
    cap = plan(get_model("llama-3.1-8b"), get_gpu("a100-80"))
    assert cap.sequences_at(4096) > 30 * cap.sequences_at(131072)


def test_weights_scale_with_dtype() -> None:
    model, gpu = get_model("llama-3.1-70b"), get_gpu("h100")
    fp16 = plan(model, gpu, gpus=4, dtype="fp16")
    awq = plan(model, gpu, gpus=4, dtype="awq")
    assert awq.weight_bytes * 4 == fp16.weight_bytes
    assert awq.kv_bytes > fp16.kv_bytes  # the saving becomes KV cache


def test_quantised_weights_do_not_quantise_the_kv_cache() -> None:
    """A distinction that is easy to miss and expensive to get wrong."""
    cap = plan(get_model("llama-3.1-8b"), get_gpu("a100-80"), dtype="awq")
    assert cap.kv_dtype == "fp16"
    assert cap.kv_per_token == get_model("llama-3.1-8b").kv_bytes_per_token("fp16")


def test_kv_dtype_can_be_overridden() -> None:
    cap = plan(get_model("llama-3.1-8b"), get_gpu("a100-80"), dtype="fp16", kv_dtype="fp8")
    assert cap.kv_per_token == get_model("llama-3.1-8b").kv_bytes_per_token("fp8")


def test_a_model_that_does_not_fit_says_so() -> None:
    cap = plan(get_model("llama-3.1-70b"), get_gpu("a100-80"), dtype="fp16")
    assert not cap.fits
    assert cap.max_sequences == 0
    assert "DOES NOT FIT" in cap.render()


def test_more_gpus_can_make_it_fit() -> None:
    model, gpu = get_model("llama-3.1-70b"), get_gpu("a100-80")
    assert not plan(model, gpu, gpus=1, dtype="fp16").fits
    assert plan(model, gpu, gpus=2, dtype="fp16").fits


def test_overhead_is_charged_per_gpu() -> None:
    """Each device has its own CUDA context and buffers."""
    model, gpu = get_model("llama-3.1-8b"), get_gpu("a100-80")
    one = plan(model, gpu, gpus=1)
    four = plan(model, gpu, gpus=4)
    assert four.overhead_bytes == 4 * one.overhead_bytes


def test_max_tokens_is_the_number_vllm_works_in() -> None:
    cap = plan(get_model("llama-3.1-8b"), get_gpu("a100-80"), context=4096)
    assert cap.max_tokens == cap.kv_bytes // cap.kv_per_token
    assert cap.max_tokens > cap.max_sequences * 4000


def test_decode_ceiling_follows_bandwidth() -> None:
    """Decode is memory-bound: twice the bandwidth, twice the ceiling."""
    model = get_model("llama-3.1-8b")
    slow = plan(model, get_gpu("a10g"))
    fast = plan(model, get_gpu("h100"))
    ratio = fast.decode_tokens_per_second() / slow.decode_tokens_per_second()
    assert abs(ratio - (3350 / 600)) < 0.01


def test_moe_decode_uses_active_parameters() -> None:
    """Memory is sized by total parameters, speed by the routed ones."""
    cap = plan(get_model("mixtral-8x7b"), get_gpu("h200"), dtype="fp16")
    dense_equivalent = 4800e9 / (46.7e9 * 2)
    assert cap.decode_tokens_per_second() > 3 * dense_equivalent


# ------------------------------------------------------------------ warnings
def test_warns_when_a_single_request_cannot_fit() -> None:
    cap = plan(get_model("llama-3.1-8b"), get_gpu("l4"), dtype="fp16", context=131072)
    assert any("not one sequence fits" in w for w in cap.warnings)


def test_warns_when_there_is_nothing_to_batch() -> None:
    cap = plan(get_model("llama-3.1-8b"), get_gpu("a10g"), dtype="fp16", context=32768)
    assert any("continuous batching has nothing to batch" in w for w in cap.warnings)


def test_warns_when_kv_heads_do_not_divide_across_gpus() -> None:
    """Tensor parallelism requires it; the server refuses to start otherwise."""
    cap = plan(get_model("qwen2.5-7b"), get_gpu("a10g"), gpus=3)
    assert any("KV heads do not divide" in w for w in cap.warnings)


def test_warns_on_reckless_utilisation() -> None:
    cap = plan(get_model("llama-3.1-8b"), get_gpu("a100-80"), utilization=0.99)
    assert any("fragmentation" in w for w in cap.warnings)


def test_warns_when_context_exceeds_the_model() -> None:
    cap = plan(get_model("gemma-2-9b"), get_gpu("a100-80"), context=32768)
    assert any("exceeds the model" in w for w in cap.warnings)


def test_moe_memory_versus_speed_is_called_out() -> None:
    cap = plan(get_model("mixtral-8x7b"), get_gpu("h200"))
    assert any("MoE" in w for w in cap.warnings)


# ----------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "kwargs", [{"dtype": "float16"}, {"gpus": 0}, {"utilization": 0}, {"utilization": 1.5}]
)
def test_invalid_inputs_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        plan(get_model("llama-3.1-8b"), get_gpu("a100-80"), **kwargs)


def test_render_and_to_dict_agree() -> None:
    cap = plan(get_model("llama-3.1-8b"), get_gpu("a100-80"), context=8192)
    data = cap.to_dict()
    assert data["fits"] is True
    assert data["max_sequences"] == cap.sequences_at(8192)
    assert "KV cache" in cap.render()
