# What it takes to serve an open model

## Why do this at all

Calling a hosted API is easier and, at low volume, cheaper. Self-hosting is
worth it when one of these is true:

| Reason | What it actually means |
|---|---|
| **Data cannot leave** | Regulatory or contractual. No amount of API terms fixes it. |
| **Volume** | Above roughly a million tokens a day, a GPU you rent by the hour starts to win. Below that it usually does not. |
| **Latency floor** | A model on your own network has no cross-region round trip and no shared-tenancy queue. |
| **The model** | A fine-tune of your own, or an open model with no hosted offering. |
| **Not being deprecated** | A model on your disk does not get retired in ninety days. |

And the reason not to: you are now operating a GPU service. The engine is the
easy part — one `docker compose up`. The rest of this document is the part
that is not.

## The three pieces

```
   client ──▶ gateway ──▶ inference engine ──▶ GPU
              (auth,       (vLLM / SGLang:     (weights +
               admission,   batching,           KV cache)
               routing,     paged attention,
               accounting)  prefix caching)
```

**The engine** — vLLM or SGLang — turns requests into tokens. It implements
continuous batching, paged attention and prefix caching, and it is very good at
it. Do not write one.

**The GPU** is the constraint, and specifically its memory. Weights are fixed;
whatever is left is KV cache; the KV cache is what sets how many requests you
can serve at once. [Capacity](02-capacity.md) is the arithmetic.

**The gateway** is everything the engine deliberately does not do:
authentication, limits across more than one replica, routing, circuit
breaking, and knowing who used what. [The gateway](03-the-gateway.md) covers
why each piece is there.

## What makes inference serving different

Four things that invalidate the instincts you would bring from serving a normal
HTTP API.

### Requests do not cost the same

A 32k-token prompt generating 2k tokens occupies a slot roughly a hundred times
longer than a 200-token prompt generating 50. Every load-balancing heuristic
that counts requests — round robin, weighted round robin, least-connections at
the TCP level — assumes a cost distribution that does not hold here. Routing on
*outstanding requests* does not need a cost model and is most of what you need.

### Two phases with opposite bottlenecks

**Prefill** processes the whole prompt at once. It is compute-bound and roughly
linear in prompt length, and it is what you wait through before the first
token.

**Decode** generates one token at a time, and each token requires reading the
entire model out of GPU memory. It is bandwidth-bound. A card with twice the
FLOPS and the same memory bandwidth decodes at about the same speed.

This is why batching works so well. Decode at batch size 1 reads the whole
model to produce one token; at batch size 32 it reads the whole model once to
produce thirty-two. The bandwidth cost is amortised, which is the single
biggest throughput lever in the stack, and it is why a server that looks idle
at low traffic can be nearly as fast at high traffic.

It is also why the two numbers people quote — "tokens per second" — are
different numbers. Per-stream speed goes *down* as batch size rises. Total
throughput goes *up*. A benchmark that reports one of them is reporting half
the story.

### Memory is allocated per token, not per request

A conventional server's memory use is roughly requests × a constant. An
inference server's is tokens-in-flight × a constant, and a single long request
can cost more than fifty short ones. This is why `--max-model-len` matters more
than almost any other flag: it is what the engine reserves per sequence, and
setting it to the model's maximum rather than to what your traffic uses can cut
your concurrency by thirty times on identical hardware.

### Responses are streams

Latency is not one number. Time to first token is what the user experiences as
"did it hear me"; inter-token latency is reading speed; total time is what a
non-streaming caller waits. They move independently and a mean over all three
is meaningless. [Benchmarking](04-benchmarking.md) is about measuring them
separately.

## The order to do things in

```bash
selfserve fit llama-3.1-8b --gpu a10g --context 8192   # 1. will it fit?
docker compose -f deploy/vllm/docker-compose.yml up    # 2. start it
selfserve probe http://localhost:8000                  # 3. is it what it claims?
selfserve bench http://localhost:8000 --sweep 1,4,16   # 4. what are the numbers?
selfserve gateway --backends http://localhost:8000     # 5. put something in front
```

Step 1 costs nothing and takes a second. Skipping it means finding out about
your capacity from a CUDA OOM several minutes into a model load, which is the
most common way a first deployment fails.

Next: [capacity](02-capacity.md).
