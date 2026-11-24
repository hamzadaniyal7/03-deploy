# The gateway

vLLM and SGLang are inference engines. They are excellent at being inference
engines, and they are explicitly not trying to be the edge of your deployment.
What they leave out is what this gateway is:

```
auth ──▶ admission ──▶ route ──▶ proxy (streaming) ──▶ account
          │             │          │
          └─ 503 ───────┴─ failover ┘ (only before the first byte)
```

```bash
selfserve gateway --backends http://vllm-a:8000,http://vllm-b:8000 \
                  --max-concurrency 64 --queue-limit 128
```

Run it in front of one replica or ten; it speaks the same OpenAI-compatible API
on both sides, so nothing downstream changes.

## Admission control

An inference server has a hard capacity: the KV cache either has room for
another sequence or it does not. Past that, a request can be refused quickly or
queued slowly.

**Queueing slowly is almost always wrong.** A queued request keeps costing its
caller time while making no progress, and by the time it reaches the front its
client has often given up — so the server spends real GPU time generating
tokens for a socket nobody is reading. A queue deep enough to absorb a spike is
a queue deep enough to serve nothing but timed-out requests during one.

So: a bounded number in flight, a bounded number waiting, and a fast 503 with
`Retry-After` past that.

```
max_concurrency = 64    # in flight across the whole deployment
queue_limit     = 128   # waiting for a slot
queue_timeout_s = 10    # a request that waited this long has already lost
```

Both bounds are needed. Concurrency alone with unlimited waiting is the
slow-queue failure above; a waiting room with no concurrency bound is not a
limit at all.

**Shedding load is a feature.** It is how a system under more load than it can
serve degrades into "some requests fail quickly" rather than "all requests fail
slowly". The first is recoverable and the second takes the caller down too.

Size `max_concurrency` from [`selfserve fit`](02-capacity.md) — roughly the
concurrent sequences one replica holds, times the replica count — and confirm
it with [`selfserve bench --sweep`](04-benchmarking.md). There is no value in
admitting load the GPUs cannot turn into tokens any faster.

## Routing: least outstanding requests

This is the one routing decision genuinely different for LLM traffic.

Round robin assumes requests cost roughly the same. For inference they differ
by two orders of magnitude — a 32k prompt generating 2k tokens against a 200
token prompt generating 50. Round robin will hand a backend its fifth expensive
request while another sits idle, because it counts requests *sent* rather than
requests still *running*.

Least-outstanding needs no cost model and no prediction: a backend busy with
expensive work has a high in-flight count by definition, so it stops receiving
traffic until it catches up.

**Ties break randomly.** Deterministic tie-breaking sends every request to the
first backend whenever the pool is idle — which is most of the time at low
traffic — so a "load balanced" pair runs entirely on one node until it gets
busy, and the second node's first request in an hour pays for a cold prefix
cache.

## Circuit breaking

Health is **passive first, active second**. The authoritative signal that a
backend is broken is that it just failed a real request; a `/health` poll
succeeding thirty seconds from now does not undo that. Failures from live
traffic open the circuit immediately, and polling is only used to close it.

After `recovery_s` the circuit goes **half-open**: the next request is a probe.
If it fails the circuit reopens, so a backend that is still broken costs one
request per window instead of all of them.

**A 503 from a backend is not a fault.** It is the server shedding load, and
counting it against health would remove a healthy backend from the pool at
exactly the moment there is not enough capacity. Rejections are counted
separately and never open a circuit.

## Failover, and when not to

| Failure | Response |
|---|---|
| Connection refused / timeout before response | retry on another backend |
| Backend returns 5xx before response | retry on another backend |
| Backend returns 503 | retry elsewhere, do not count against its health |
| Backend returns 4xx | relay to the client, never retry |
| Stream fails after the first byte | truncate and report; **never** retry |

The last row is the subtle one. Once the first byte of a 200 response has
reached the client, the status is committed and the caller holds partial
output. "Retrying" would splice two different completions together with no way
for the caller to tell. So retries happen strictly before the first byte, and
after that a failure is a truncated stream and is reported as one.

A 4xx will fail identically on every backend; retrying it just multiplies the
work and delays an error the caller needs now.

## Streaming is a passthrough

Chunks are forwarded as they arrive, never buffered. Collecting the response
and sending it at the end would be simpler and would destroy the only property
a streaming endpoint has — the user would wait for the entire generation before
seeing anything, turning a 200ms time-to-first-token into fifteen seconds.

This is also the most common way a *working* deployment ends up with broken
streaming: a reverse proxy in front with buffering left on. `selfserve probe`
checks for it explicitly, because from the outside the endpoint looks fine.

## Accounting

Token counts are sniffed out of the response as it streams past — forwarded
first, inspected second, so accounting never sits between the backend and the
user. The numbers are the server's own usage block rather than an estimate from
chunk counts, which is what makes them usable for cost attribution.

```bash
curl localhost:8080/status    # backends, in-flight, queue depth, usage
curl localhost:8080/metrics   # totals plus the recent request log
```

`/status` is what an autoscaler should read. **Scaling inference on request
count is a trap**: a replica with four 32k requests is fully loaded while one
with forty 200-token requests is idle, and a request-count autoscaler cannot
tell them apart. In-flight counts and queue depth can.

## What it deliberately does not do

**TLS and real identity.** It speaks plain HTTP and checks one shared bearer
token. That is right for something behind an ingress that already terminates
TLS and authenticates, and not enough to expose directly.

**Rate limiting per tenant.** The accounting is there; the enforcement is not.
It belongs in a gateway that already knows who your tenants are.

**Request transformation.** It does not rewrite prompts, inject system
messages, or filter content. A proxy that modifies requests is a proxy whose
behaviour you have to reason about on every incident.

Next: [benchmarking](04-benchmarking.md).
