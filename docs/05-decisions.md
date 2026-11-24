# Decisions

What was chosen, what it cost, and what was rejected. The rejected options are
the useful part.

---

## ADR 1: Do not write an inference engine

**Decision.** Use vLLM or SGLang. This project is the arithmetic around them,
the gateway in front of them, and the measurement of them.

**Why.** Continuous batching, paged attention, prefix caching and the CUDA
kernels underneath are years of specialised work, and both projects do them
well. Reimplementing any of it would produce something slower and would teach
the wrong lesson: the hard part of serving a model is not the forward pass, it
is everything that decides *which* forward passes happen and when.

**Rejected: a minimal engine for teaching.** Tempting, and it would be a toy.
The capacity arithmetic in `capacity.py` is the part of an engine worth
understanding, and it is seventy lines.

**Cost.** This repo is useless without one of them running. The mock exists so
that everything *else* can be developed and tested without a GPU.

---

## ADR 2: No dependencies

**Decision.** The client speaks HTTP with `urllib` and parses SSE by splitting
on newlines. The gateway and the mock run on `http.server`. The runtime
dependency list is empty.

**Why.** Two reasons, and the second is the real one.

The first: `openai` is a large dependency for POSTing JSON and splitting lines,
and `fastapi` plus `uvicorn` is a lot of machinery for a proxy with six routes.

The second: **a benchmark that depends on a client library measures that
library.** Connection pooling, retry behaviour, JSON parsing and buffering all
sit between the measurement and the server. When a number looks wrong, "is it
the server or is it the client?" is a question you cannot answer without
reading someone else's HTTP stack. With `urllib` the whole path is visible and
`time.perf_counter()` is read the instant a chunk arrives.

**Rejected: `httpx` + `fastapi` + `uvicorn`.** Better ergonomics, async
throughout, and genuinely the right answer for a gateway serving serious
traffic — which is why `[asgi]` is an extra. For a reference implementation the
stdlib is more legible.

**Cost.** The gateway is threads rather than async, so it is not what you would
run in front of a large fleet. It handles tens of concurrent streams fine,
which is what a single-node deployment needs.

---

## ADR 3: Least outstanding requests, not round robin

**Decision.** Route to the backend with the fewest requests currently running,
breaking ties randomly.

**Why.** LLM request costs differ by two orders of magnitude. Round robin
counts requests *sent*; what matters is requests still *running*, and a backend
busy with expensive work has a high in-flight count by definition. No cost
model, no prediction, no configuration.

Random tie-breaking matters more than it sounds: deterministic ordering sends
everything to the first backend whenever the pool is idle, which at low traffic
is most of the time.

**Rejected: round robin.** Correct for uniform costs. These are not uniform.

**Rejected: routing by estimated cost (prompt length).** Prompt length predicts
prefill but not output length, which is the larger term and unknowable in
advance. An estimate that is wrong half the time is worse than counting what is
actually there.

**Rejected: session affinity for prefix-cache hits.** Genuinely valuable —
routing a conversation back to the replica that already holds its prefix is a
real win. It also conflicts with load balancing, needs a session key the
OpenAI API does not have, and degrades badly when one replica goes down. Worth
doing deliberately, not by default.

**Cost.** In-flight count is maintained by the gateway, so it is blind to
traffic reaching a backend by any other path.

---

## ADR 4: Shed load rather than queue it

**Decision.** Bounded concurrency, a bounded waiting room, and a fast 503 with
`Retry-After` past that.

**Why.** A queued request keeps costing its caller time while making no
progress. By the time it is served its client has often given up, so the GPU
generates tokens for a socket nobody reads. A queue deep enough to absorb a
spike is deep enough to serve nothing but timed-out requests during one.

Refusing quickly degrades into "some requests fail fast", which callers can
retry or route around. Queueing degrades into "everything is slow", which takes
the caller down too.

**Rejected: an unbounded queue.** The default in most proxies, and the thing
that turns a busy minute into a ten-minute outage.

**Rejected: no limit at all, let the engine's scheduler handle it.** The engine
bounds what one process runs concurrently. It cannot know you have two of them,
and it cannot refuse a request before it has paid to receive it.

**Cost.** Two numbers to size. Both come out of `selfserve fit` and
`selfserve bench --sweep`.

---

## ADR 5: Never retry after the first byte

**Decision.** Failover happens strictly before any response byte reaches the
client. After that, a failure is a truncated stream and is reported as one.

**Why.** Once the first chunk is sent, the 200 status is committed and the
caller holds partial output. Retrying would splice two different completions
together and the caller would have no way to detect it. A visibly truncated
response is a much better failure than a silently corrupted one.

**Rejected: buffer the whole response so retries stay possible.** This would
make retries safe and destroy streaming — the user waits for the full
generation before seeing anything. Trading the endpoint's one defining property
for a rarely-exercised recovery path is not a good trade.

**Cost.** A backend that dies mid-generation costs the caller that request.
Clients should be prepared for truncation, which they have to be regardless.

---

## ADR 6: The mock models contention

**Decision.** The simulated backend's latency depends on prompt length and on
how many requests are already in flight, and it refuses past a capacity limit.

**Why.** A mock with fixed latency makes every benchmark produce a straight
line, and the shape of the curve as concurrency rises is the entire reason to
benchmark an inference server. Modelling contention reproduces the real
trade-off — more concurrency means higher throughput and worse per-user
latency, up to a cliff — so the harness, the report and the sweep analysis are
exercised against something that behaves like the thing they will meet.

The absolute numbers are invented. The shape is not, and the shape is what you
tune against.

**Rejected: record and replay real server responses.** More faithful for one
hardware configuration, useless for any other, and it cannot respond to load at
all.

**Rejected: no mock, integration tests only.** Would mean no tests without a
GPU, which means no tests in CI, which means no tests.

**Cost.** It is a simulator, and nothing it reports predicts your hardware. The
module docstring says so, this says so, and `selfserve mock` prints it on
startup — because a plausible-looking number from a simulator is the most
dangerous output this repo can produce.

---

## ADR 7: Programmatic capacity, not profiling

**Decision.** `selfserve fit` computes from architecture and spec sheets. It
never loads a model.

**Why.** The answer is needed *before* downloading 140GB of weights. The
calculation is exact for weights and KV cache — those are determined by the
architecture — and the only estimated term is a couple of GB of overhead, which
is small and errs the safe way.

The alternative is vLLM's own behaviour: download, load, profile, and then fail
with a CUDA OOM several minutes in. That is a fine way to confirm a plan and a
terrible way to make one.

**Rejected: parse vLLM's startup logs.** Accurate, and requires everything you
were trying to decide about to already be running.

**Cost.** The overhead figure is approximate, and real usage varies with batch
size and engine version. It is called out as its own line in the output rather
than folded invisibly into the result.

---

## ADR 8: Closed-loop load generation

**Decision.** `concurrency` workers, each sending the next request when the
previous finishes.

**Why.** It measures what a fixed pool of callers experiences, which is the
question most people actually have ("how many concurrent users can this
serve?"), and it is stable enough to compare across runs.

**Rejected: open-loop arrival at a fixed rate.** More faithful to real traffic
and the only way to find the cliff — a closed loop backs off automatically when
the server slows, so it approaches saturation rather than passing through it.
It is on the roadmap. The decision here is to implement one properly and
document the difference, rather than implement both badly and leave the reader
unsure which they ran.

**Cost.** The sweep answers "what do N concurrent users see", not "what happens
at N requests per second". Those differ once the server is saturated, and a
great many published benchmarks conflate them.

---

## ADR 9: Threads, not asyncio

**Decision.** One thread per in-flight request, in both the benchmark and the
gateway.

**Why.** The work is almost entirely waiting on a socket, so both models serve.
Threads win on legibility: a benchmark whose own concurrency model is exotic
invites the question of whether it is measuring the server or itself, and the
most boring possible answer is the right one underneath a measurement.

**Rejected: asyncio throughout.** Scales to far more concurrent connections and
is what a production gateway should use. At the tens-to-low-hundreds of streams
a single-node deployment handles, the difference is not measurable, and the
stdlib async HTTP story is much worse than the sync one.

**Cost.** Thousands of concurrent streams would need the `[asgi]` extra and a
real server. Said plainly in the [roadmap](06-roadmap.md) rather than
discovered under load.
