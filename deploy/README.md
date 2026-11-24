# Deploying

Three compose files, in increasing order of how much they assume:

| File | What it runs | When |
|---|---|---|
| `vllm/docker-compose.yml` | one vLLM replica | the normal starting point |
| `sglang/docker-compose.yml` | one SGLang replica, same API | comparing engines |
| `docker-compose.gateway.yml` | two replicas behind the gateway | more than one GPU |

All three need the NVIDIA Container Toolkit. None of them need anything from
this repo installed on the host except `selfserve` itself for `fit`, `probe`
and `bench`.

## Do the arithmetic first

```bash
selfserve fit llama-3.1-8b --gpu a10g --context 8192
```

The two flags that decide whether a deployment works are
`--gpu-memory-utilization` and `--max-model-len`, and both come straight out of
that output. Guessing them does not fail fast: vLLM downloads the weights,
loads them, profiles, and *then* dies with a CUDA OOM several minutes in.

The single most common mistake is leaving `--max-model-len` at the model's
maximum. Llama 3.1 supports 128k context; serving at 128k instead of the 8k
your traffic actually uses costs you about thirty times the concurrency, on the
same card, for no benefit to anyone.

## Then check the server is what it claims

```bash
selfserve probe http://localhost:8000
```

"OpenAI-compatible" is a spectrum. This checks the parts that silently differ:
whether usage is reported in streaming responses, whether a health endpoint
exists, and whether chunks actually arrive incrementally rather than being
buffered somewhere into one burst at the end.

## Then measure it

```bash
selfserve bench http://localhost:8000 --sweep 1,4,16,64
```

Find the concurrency where throughput stops rising and latency starts running
away. That is your operating point, and it is also what `SELFSERVE_MAX_CONCURRENCY`
should be set to — there is no value in admitting load the GPU cannot turn into
tokens any faster.

## Environment variables

Every compose file reads these, with the defaults shown:

| Variable | Default | Notes |
|---|---|---|
| `MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | any HF repo the engine supports |
| `MAX_MODEL_LEN` | `8192` | the capacity lever — see above |
| `GPU_MEMORY_UTILIZATION` | `0.90` | vLLM only; `MEM_FRACTION` for SGLang |
| `TP_SIZE` | `1` | tensor parallel degree |
| `GPU_COUNT` | `1` | GPUs given to the container |
| `HF_HOME` | `~/.cache/huggingface` | host weight cache; mount it or re-download |
| `HUGGING_FACE_HUB_TOKEN` | *(unset)* | required for gated repos |
| `SELFSERVE_MAX_CONCURRENCY` | `64` | gateway, across all replicas |
| `SELFSERVE_API_KEY` | *(unset)* | if set, clients must send it |

## What is not here

**TLS, and a real identity system.** The gateway speaks plain HTTP and checks a
single shared bearer token. That is the right amount of auth for something
behind an ingress that already does TLS and real authentication, and it is not
enough to put on the public internet by itself.

**Autoscaling.** Scaling inference on request count is a trap — a replica with
four 32k requests is fully loaded while one with forty 200-token requests is
idle, and they look the same to a request-count autoscaler. `GET /status`
exposes in-flight counts and queue depth, which is what a sane scaler should
read instead.

**Kubernetes manifests.** The compose files say what matters; translating them
is mechanical, and a half-right Helm chart is worse than none.
