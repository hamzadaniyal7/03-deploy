# Benchmarking

```bash
selfserve bench http://localhost:8000 --sweep 1,4,16,64
```

```
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

*(Numbers from the bundled simulator, not from a GPU — they illustrate the
shape. Point it at a real server for numbers about your hardware.)*

That table is the whole point. Throughput rises 7x; the latency a user
experiences gets 5x worse; past 16 the throughput gain has nearly stopped while
the latency cost has not. Somewhere between 16 and 32 is your operating point,
and no single-number benchmark can tell you where.

## The three latencies

A streaming endpoint has three, and they move independently.

| Metric | What the user feels | What it scales with |
|---|---|---|
| **TTFT** | "did it hear me?" | prompt length, queue depth ahead of you |
| **ITL** | reading speed | batch size — more concurrency, slower stream |
| **end-to-end** | what a non-streaming caller waits | both |

A chat UI lives and dies on TTFT. A batch summarisation job does not care about
TTFT at all and cares entirely about total throughput. Tuning for one actively
harms the other, so "is it fast?" is not a question with an answer until you
say which.

## Never report a mean alone

Inference latency is long-tailed by construction: a request that queued behind
a 32k prefill waits an order of magnitude longer than one that did not. The
mean sits in a gap where almost no request landed.

Five percent of requests at 100x the latency leaves p50 untouched, drags the
mean to a value nobody experienced, and shows up properly only in p95 and p99 —
which is exactly what users complain about. Two servers with identical means
can have p99s a factor of ten apart.

## Separate server throughput from user latency

They move in opposite directions as batch size grows. A report that collapses
them into one "performance" number cannot show the trade-off that is the entire
subject of tuning.

* **output tokens/s** is the server's decode throughput — the number that pays
  for the GPU.
* **per-stream tok/s** is one user's reading speed. It goes *down* as the
  server gets busier, and that is correct behaviour, not a regression.

## Methodology, and the ways it goes wrong

### Warm up, and throw the warm-up away

The first request to a freshly started server pays for CUDA graph capture,
weight paging and an empty prefix cache. It can be ten times slower than the
second. Including it in a percentile reports a latency problem that does not
exist. `--warmup` requests are sent and discarded — discarded rather than
merely excluded, because they also bring the scheduler to a steady state.

### Build prompts out of words

A prompt of repeated characters tokenises to a fraction of its apparent length,
so the prefill cost you were trying to measure never happens. The generator
here uses real words at roughly 0.75 tokens each. The exact count matters less
than it being *consistent between runs*, so a number measured today is
comparable with one measured after a config change.

### Closed loop versus open loop

This is a **closed-loop** generator: `concurrency` workers each send a request,
wait for the whole response, and send the next. It measures what a fixed pool
of callers experiences, and its results are stable enough to compare across
runs.

It has a property worth knowing: **when the server slows down, the offered load
drops with it**, so the server is never pushed past the point where it stops
keeping up. An **open-loop** generator sends at a fixed rate regardless, which
is what real traffic does and which is how you find the cliff rather than
approaching it asymptotically.

A great many published benchmark numbers are closed-loop results described as
though they were open-loop. Open-loop arrival is on the
[roadmap](06-roadmap.md); until then, read the sweep as "what N concurrent
users experience", not "what happens at N requests per second".

### Take token counts from the server

With `stream_options.include_usage` honoured, the usage block gives real token
counts. Without it, the client counts chunks — close for most tokenizers, but
an estimate, and `selfserve probe` tells you which one you are getting.

### Hold prompt and output length fixed

Throughput in tokens/second is meaningless if the token counts move between
runs. `--prompt-tokens` and `--max-tokens` are fixed per run for this reason,
and comparing two runs with different values compares nothing.

## Reading the result

**Rejections are not errors.** A 503 is the system working as designed; a 500
is a bug; a timeout is a request that consumed capacity and produced nothing.
They are counted separately because the three have completely different fixes,
and lumping them together hides which you have.

**Where load shedding starts** is the most actionable line in the output. That
is your real capacity, and it is what `--max-concurrency` on the gateway should
be set near.

**If `stuck` and `budget` numbers are all zero** across a whole sweep, the
levels are too low to find anything. Push until something gives; a benchmark
that never reaches a limit has not measured one.

## Comparing engines

The vLLM and SGLang compose files in [`deploy/`](../deploy/) serve the same
model behind the same API, so comparing them is one URL change:

```bash
selfserve bench http://localhost:8000 --sweep 1,8,32 --no-save   # vLLM
selfserve bench http://localhost:8001 --sweep 1,8,32 --no-save   # SGLang
```

Which wins depends on your workload — prompt lengths, output lengths, how much
prefix is shared — rather than on a blog post. That is the reason to own the
harness.

Next: [the decisions](05-decisions.md).
