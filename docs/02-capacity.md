# Capacity

The question is never "does the model fit". An 8B model in fp16 is 16GB and
obviously fits on an 80GB card. The question is **how much KV cache is left
afterwards**, because that is what decides how many requests you can serve at
the same time.

```bash
selfserve fit llama-3.1-8b --gpu a100-80 --context 4096
```

```
Llama 3.1 8B Instruct @ fp16
1x NVIDIA A100 80GB  --  utilisation 0.90

  total memory       80.0 GB
  usable             72.0 GB   (x0.90)
  weights            15.0 GB
  overhead            2.0 GB
  KV cache           55.0 GB

  KV per token        128 KB   (2 x 32 layers x 8 kv_heads x 128 dim x 2B)
  KV tokens total    450,911

  concurrent sequences by context length:
        2k     220
        4k     110  <- configured
        8k      55
       16k      27
       32k      13
      128k       3

  decode ceiling         127 tok/s   single stream, memory-bound estimate
```

## Where the memory goes

```
GPU memory = weights + KV cache + overhead
```

**Weights** are `parameters × bytes-per-parameter`, and they do not move. fp16
is 2 bytes; fp8 and int8 are 1; 4-bit schemes like AWQ and GPTQ are about 0.5.

**Overhead** is the CUDA context, activation buffers and allocator
fragmentation — a couple of GB, charged per GPU because every device has its
own.

**KV cache** is whatever is left, and it is the interesting number.
`--gpu-memory-utilization` (vLLM's default 0.9) caps what the process may take
in total; the KV cache gets the remainder after weights and overhead.

## The KV cache formula

Per token, per sequence:

```
2 × layers × kv_heads × head_dim × bytes_per_element
```

The `2` is keys and values. For Llama 3.1 8B in fp16:

```
2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KB per token
```

Everything in it is linear, which is why a 128k-context request is not "a bit
more memory" than a 4k one — it is thirty-two times more.

### kv_heads is the number people get wrong

Modern models use **grouped-query attention**: many query heads share one
key/value head. Llama 3.1 8B has 32 attention heads and 8 KV heads. Sizing the
cache from the attention-head count overestimates it by 4x — the difference
between "fits comfortably" and "OOM on the third concurrent request".

Qwen2.5 7B is more extreme still: 28 query heads, 4 KV heads, a 7x ratio.

If you are reading a `config.json` yourself, the field is
`num_key_value_heads`, and when it is absent the model predates GQA and the
answer is `num_attention_heads`.

### head_dim is usually but not always hidden_size / heads

Gemma 2 9B has `hidden_size` 3584 and 16 attention heads, which would give 224
— but its actual `head_dim` is 256. Check the field before deriving it.

## What this buys you

Two numbers come out, and the second is the one to plan with.

**Concurrent sequences at full context** assumes every request runs to the
context limit. It is the pessimistic bound and it is what you want for a
worst-case guarantee.

**Total KV tokens in flight** is the number vLLM actually schedules against,
and it is the honest planning number: twenty sequences averaging 2k tokens cost
the same as one at 40k. Real traffic is nowhere near uniform, so a deployment
sized on the first number is usually leaving half the card unused.

## The flag that matters most

`--max-model-len`. The engine reserves KV for this many tokens per sequence.

| `--max-model-len` | Concurrent sequences (8B, A100 80GB) |
|---|---|
| 4096 | 110 |
| 8192 | 55 |
| 32768 | 13 |
| 131072 | 3 |

Same model, same card, a 37x difference. Llama 3.1 supports 128k; serving at
128k when your traffic uses 4k costs you almost all of your concurrency for the
benefit of nobody. Set it to what you serve.

## Quantisation

Quantising weights frees memory that becomes KV cache, so the benefit is larger
than the weight saving alone suggests — on a card where the model only just
fits, going from fp16 to 4-bit can take you from two concurrent sequences to
thirty.

```bash
selfserve fit llama-3.1-70b --gpu a100-80 --gpus 2 --dtype awq --context 32768
```

**Weight quantisation does not quantise the KV cache.** `--dtype awq` with no
further configuration leaves KV at fp16, and on a long-context deployment the
KV cache is the larger of the two. Quantising it as well is a separate flag,
and `selfserve fit --kv-dtype fp8` shows what it would buy before you decide
whether the quality cost is worth it.

## Multiple GPUs

Tensor parallelism splits each layer across GPUs; it is the default choice and
it needs `kv_heads` to be divisible by the parallel degree. A model with 8 KV
heads works on 1, 2, 4 or 8 GPUs and not on 3 — `selfserve fit --gpus 3` warns
about this rather than letting the engine fail at startup.

Memory scales with GPU count; so does overhead, since each device carries its
own context.

## Decode speed

The reported ceiling is `memory bandwidth / bytes read per token`. Generating
one token means reading every weight the forward pass touches out of HBM, so
that ratio is the hard upper bound for a single stream.

It is an upper bound and real servers land well under it — it ignores attention
over the KV cache, kernel launch overhead and sampling. Use it for the order of
magnitude, and for the comparison it makes obvious: **batching is what turns a
bandwidth-bound decode into a compute-bound one.** One stream reads the whole
model to make one token; thirty-two streams read it once to make thirty-two.

For MoE models the figure uses active parameters, because only the routed
experts are read. Mixtral 8x7B occupies memory like a 47B model and decodes
like a 13B one, and a single number cannot describe both.

## Anything not in the table

```bash
selfserve fit --hf-config path/to/config.json --params-b 8.03 --gpu h100
```

Reads the real architecture off the model's own config. Use this for anything
you actually deploy; the built-in table is a convenience for the common cases
and a snapshot of what was true when it was written.

Next: [the gateway](03-the-gateway.md).
