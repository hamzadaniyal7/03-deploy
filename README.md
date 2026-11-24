# 03-deploy — serve an open model yourself, and know the numbers

Three questions stand between "I downloaded a model" and "I am serving it":
**will it fit, what happens under load, and what goes in front of it.** This
answers all three.

The first one costs nothing and takes a second:

```
$ selfserve fit llama-3.1-8b --gpu a10g --context 8192

Llama 3.1 8B Instruct @ fp16
1x NVIDIA A10G  --  utilisation 0.90

  total memory       24.0 GB
  usable             21.6 GB   (x0.90)
  weights            15.0 GB
  overhead            2.0 GB
  KV cache            4.6 GB

  KV per token        128 KB   (2 x 32 layers x 8 kv_heads x 128 dim x 2B)
  KV tokens total     38,035

  concurrent sequences by context length:
        2k      18
        4k       9
        8k       4  <- configured
       16k       2
       32k       1
      128k       0

  decode ceiling          37 tok/s   single stream, memory-bound estimate
```

An 8B model "fits on a 24GB card" — and serves **four** concurrent requests at
8k context, one at 32k, and none at the 128k the model advertises. That table
is the deployment decision, and almost nobody computes it before starting the
server. The usual way to find out is a CUDA OOM several minutes into a model
load.

No model is downloaded and no GPU is touched to produce it. It is arithmetic —
weights, overhead, and what is left for the KV cache — and it lives in
[one readable module](src/selfserve/capacity.py) that shows its working rather
than just its answer.

---

## Quickstart

```bash
git clone https://github.com/hamzadaniyal7/03-deploy && cd 03-deploy
python -m venv .venv && source .venv/bin/activate     # Python 3.11+
pip install -e '.[dev]'

selfserve fit --list                                  # known models and GPUs
selfserve fit llama-3.1-8b --gpu a100-80 --context 8192
```

`fit` needs nothing. For the rest, a simulated backend ships with the repo, so
**everything below runs with no GPU and no network**:

```bash
selfserve mock --port 8000 &                          # a fake vLLM
selfserve probe http://localhost:8000                 # is it what it claims?
selfserve bench http://localhost:8000 --sweep 1,4,16,32
selfserve gateway --backends http://localhost:8000    # auth, limits, routing
```

Against a real server, drop the `mock` line and point at port 8000 of
[`deploy/vllm/docker-compose.yml`](deploy/vllm/docker-compose.yml).

---

## The sweep is the point

```
$ selfserve bench http://localhost:8000 --sweep 1,4,16,32

   conc     req/s   out tok/s   TTFT p50   TTFT p99   ITL p50   ITL p99   503s
  ----------------------------------------------------------------------------
      1      8.35       200.3        37m        42m      3.3m      3.9m      0
      4     26.87       645.0        49m        52m      4.0m      4.8m      0
     16     55.54      1333.0        84m       185m      7.0m      8.5m      0
     32     59.18      1420.4        95m       209m      8.0m     10.8m     91

  peak throughput at concurrency 32: 1420 output tok/s, TTFT p99 209ms
  versus concurrency 1: 7.1x the throughput for 5.0x the TTFT p99
  load shedding begins at concurrency 32 (91 rejected)
```

Throughput rises 7x. The latency a user feels gets 5x worse. Past 16 the
throughput gain has nearly stopped and the latency cost has not. **Somewhere
between 16 and 32 is your operating point**, and no single-number benchmark can
tell you where it is.

*(That run is against the bundled simulator — it illustrates the shape, not
your hardware. The simulator models contention deliberately: prefill cost grows
with prompt length and decode slows as the batch grows, because a mock with
fixed latency produces a straight line and the curve is the entire reason to
benchmark.)*

Two rules the report follows:

* **Never a mean alone.** Inference latency is long-tailed by construction.
  Five percent of requests at 100x leaves p50 untouched and drags the mean
  somewhere no request actually landed. Two servers with identical means can
  have p99s a factor of ten apart.
* **Server throughput and user latency are separate numbers.** They move in
  opposite directions as batch size grows. Collapsing them into one
  "performance" figure hides the only trade-off that matters.

---

## Check the server is what it claims

"OpenAI-compatible" is a spectrum. Every stack claims it; they differ in the
optional parts, and the differences surface only once something depends on them.

```
$ selfserve probe http://localhost:8000

  [ok  ] health endpoint              /health -> 200
  [ok  ] GET /v1/models               mock-8b
  [ok  ] non-streaming completion     8 tokens in 0.16s
  [ok  ] streaming completion         24 chunks, first at 0.044s
  [ok  ] usage in stream              token counts came from the server
  [ok  ] chunks arrive incrementally  378ms between first and last chunk
  [ok  ] round trip                   0.44s

  usable
```

That fifth line decides whether all your downstream accounting is measured or
estimated. The sixth catches a reverse proxy with buffering left on — the
endpoint works, streaming does not, and from the outside it looks fine until a
user complains the UI shows nothing for ten seconds.

---

## What goes in front

vLLM and SGLang are inference engines, and deliberately not the edge of your
deployment: no auth, no notion of more than one replica, no admission control
beyond their own scheduler queue, no per-caller accounting.

```bash
selfserve gateway --backends http://vllm-a:8000,http://vllm-b:8000 \
                  --max-concurrency 64 --queue-limit 128
```

```
auth ──▶ admission ──▶ route ──▶ proxy (streaming) ──▶ account
          │             │          │
          └─ 503 ───────┴─ failover ┘ (only before the first byte)
```

| Piece | Decision, and why it is not the obvious one |
|---|---|
| **Admission** | Bounded in flight, bounded queue, fast 503 with `Retry-After`. A queue deep enough to absorb a spike is deep enough to serve nothing but timed-out requests during one. |
| **Routing** | Least *outstanding* requests, not round robin. LLM request costs differ by 100x, so counting requests sent is not counting requests running. Ties break randomly — deterministic order sends everything to node one whenever the pool is idle. |
| **Circuit breaking** | Opened by live traffic failures, closed by health polls, half-open on recovery. A backend's 503 never counts against it: that is load shedding, and penalising it removes capacity exactly when there is none. |
| **Failover** | Only before the first response byte. After that the 200 is committed and the caller holds partial output, so a "retry" would splice two completions together invisibly. |
| **Streaming** | Passthrough, never buffered. Buffering turns a 200ms time-to-first-token into the full generation time. |
| **Accounting** | Token counts sniffed from the stream as it passes — the server's own usage block, not an estimate. Forwarded first, inspected second. |

```bash
curl localhost:8080/status    # backends, in-flight, queue depth, usage
```

`/status` is what an autoscaler should read. Scaling inference on request count
is a trap: a replica with four 32k requests is fully loaded while one with
forty 200-token requests is idle, and a request-count scaler cannot tell them
apart.

---

## Commands

```bash
selfserve fit <model> [--gpu G] [--gpus N] [--dtype D] [--context C] [--json]
selfserve fit --hf-config path/to/config.json --params-b 8.03    # any model
selfserve probe <url>                        # compatibility checks
selfserve bench <url> [-c N] [-n N] [--sweep 1,4,16] [--json]
selfserve gateway [--backends a,b] [--max-concurrency N]
selfserve mock [--port 8000] [--capacity 32] # a simulated backend
selfserve init                               # write a starter selfserve.toml
```

`fit` exits non-zero when the model does not fit, so a deploy script can check
before starting anything.

As a library:

```python
from selfserve import plan, get_model, get_gpu

capacity = plan(get_model("llama-3.1-70b"), get_gpu("h100"), gpus=4,
                dtype="awq", context=32768)
print(capacity.max_sequences, capacity.max_tokens)
print(capacity.warnings)        # TP divisibility, context overruns, MoE notes
```

---

## Deploying for real

[`deploy/`](deploy/) has three compose files: one vLLM replica, one SGLang
replica, and two replicas behind the gateway. All three are commented with the
flags that matter and why.

The two that decide whether it works are `--gpu-memory-utilization` and
`--max-model-len`, and both come straight out of `selfserve fit`. The single
most common mistake is leaving `--max-model-len` at the model's maximum:

| `--max-model-len` | Concurrent sequences (8B, A100 80GB) |
|---|---|
| 4096 | 110 |
| 8192 | 55 |
| 32768 | 13 |
| 131072 | 3 |

Same model, same card, a 37x difference for serving a context length your
traffic never uses.

---

## Development

```bash
pytest                 # 183 tests, no GPU and no API key, ~40s
ruff check . && ruff format --check .
```

The suite runs real HTTP over real sockets against the simulated backend —
actual SSE framing, actual concurrency, actual timing — rather than mocking the
layer under test. That is why it takes forty seconds instead of two, and it is
why it catches things mocks do not: the SSE chunked-framing bug, the
post-`[DONE]` disconnect being miscounted as a truncated stream, a backend's
503 opening its circuit breaker. CI runs it on Python 3.11, 3.12 and 3.13.

**No runtime dependencies.** The client speaks HTTP with `urllib` and parses
SSE by splitting on newlines; the gateway and mock run on `http.server`. Partly
because `openai` is a lot of dependency for POSTing JSON — but mostly because
a benchmark that depends on a client library measures that library's connection
pooling, retries and buffering alongside the server, and when a number looks
wrong you cannot tell which.

---

## Reading order

| Doc | What it covers |
|---|---|
| [docs/01-what-it-takes-to-serve.md](docs/01-what-it-takes-to-serve.md) | Why self-host, the three pieces, and the four ways inference differs from normal serving |
| [docs/02-capacity.md](docs/02-capacity.md) | The KV cache formula, GQA, quantisation, multi-GPU, and the flag that matters most |
| [docs/03-the-gateway.md](docs/03-the-gateway.md) | Admission, routing, circuit breaking, failover rules |
| [docs/04-benchmarking.md](docs/04-benchmarking.md) | The three latencies, methodology, and how benchmarks go wrong |
| [docs/05-decisions.md](docs/05-decisions.md) | ADRs — the trade-offs, including the ones rejected |
| [docs/06-roadmap.md](docs/06-roadmap.md) | Known limits and what comes next |
| [deploy/README.md](deploy/README.md) | The compose files and their environment variables |

## Where this sits

Project 3 of 3 in a build-it-yourself AI-engineering series:

| | Project | What it teaches |
|---|---|---|
| 01 | [rag](https://github.com/hamzadaniyal7/01-rag) | Retrieval, grounding, and citations you can verify |
| 02 | [agents](https://github.com/hamzadaniyal7/02-agents) | Loop engineering — plan → act → check over many steps |
| **03** | **deploy** *(this repo)* | Serving an open model yourself with vLLM or SGLang |

## License

MIT — see [LICENSE](LICENSE).
