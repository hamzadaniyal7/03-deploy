# Limits, and what comes next

## Known limits

**The gateway is threads on `http.server`.** Fine for the tens to low hundreds
of concurrent streams a single-node deployment handles; not what you put in
front of a large fleet. The `[asgi]` extra exists for that, and the handler
would need rewriting against it.

**No TLS, and one shared bearer token.** Correct behind an ingress that already
terminates TLS and authenticates. Not enough to expose directly, and no amount
of configuration here makes it so.

**Load generation is closed-loop only.** The sweep answers "what do N
concurrent users see", not "what happens at N requests per second". Once the
server saturates these differ, and the second is how you find the cliff. See
[ADR 8](05-decisions.md#adr-8-closed-loop-load-generation).

**The capacity model assumes dense transformer attention.** Multi-head latent
attention (DeepSeek-V2 and successors) compresses the KV cache in a way the
`2 × layers × kv_heads × head_dim` formula does not describe, and sliding-window
models (Mistral's older configs, Gemma's alternating layers) cap KV growth
below the context length. Both would be overestimated here.

**Overhead is a flat 2GB per GPU.** It actually scales with batch size and
hidden dimension. Flat is close enough for sizing and errs the safe way, but it
is the one estimated term in an otherwise exact calculation.

**The model table is a snapshot.** Architecture facts read from published
configs at one point in time. `--hf-config` reads the real thing and is what
you should use for anything you deploy.

**The mock is a simulator.** Its numbers are invented; only the shape of its
response to load is meaningful. Nothing it reports predicts your hardware.

**No per-tenant rate limiting.** The accounting exists, the enforcement does
not.

**Single model per deployment.** The gateway routes between replicas of one
model. Routing *by* model — several models behind one endpoint — is a different
and larger piece of work.

## What would come next, in order

**1. Open-loop load generation.** `--rate 50` alongside `--concurrency`. The
single most valuable addition, because it is the only way to measure behaviour
past saturation rather than asymptotically approaching it.

**2. An async gateway on the `[asgi]` extra.** Same logic, same tests, uvicorn
underneath. The admission control, routing and circuit breaking are already
independent of the transport; only the handler is coupled to `http.server`.

**3. Prefix-aware routing.** Hash a request's leading tokens and prefer the
replica that served that prefix most recently, falling back to least-outstanding
when the hit is cold or that replica is loaded. A large win for chat workloads
with long shared system prompts, and it needs care not to undo load balancing.

**4. MLA and sliding-window capacity models.** `ModelSpec` would grow an
attention-kind field, and `kv_bytes_per_token` would branch on it. Mechanical,
and it needs correct per-architecture formulas rather than a guess.

**5. Prometheus metrics.** `/metrics` currently returns JSON. The data is
already there; it needs the text exposition format and a histogram for TTFT.

**6. Speculative decoding in the capacity model.** Draft models change both the
memory picture (two models resident) and the throughput picture (accepted
tokens per forward pass). Currently not modelled at all.

**7. A recorded real-server trace in the repo.** Every number in the README is
from the simulator, because those are the ones anyone can reproduce without a
GPU. A committed capture of a real vLLM run would let CI detect changes in the
report maths against realistic data.

## What is deliberately not planned

**A model-serving engine.** See
[ADR 1](05-decisions.md#adr-1-do-not-write-an-inference-engine).

**Kubernetes manifests and a Helm chart.** The compose files say what matters
and translating them is mechanical. A half-right chart is worse than none,
because people run it.

**Autoscaling.** `/status` exposes in-flight counts and queue depth, which is
what a scaler should read. Writing the scaler means picking a platform, and the
useful contribution is the signal rather than one platform's consumer of it.

**A web dashboard.** `/status` is JSON and `jq` exists.
