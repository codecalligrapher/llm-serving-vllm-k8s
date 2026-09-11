# llm-serving-vllm-k8s

## Links to the blog-posts:
[1. Building a multi-stage Docker image for locally serving an LLM](https://aadi-blogs.web.app/blog/docker-llm/)  
[2. Spinning up a simple k3s to manage a local LLM Docker Container](https://aadi-blogs.web.app/blog/intro-to-k3s/)


Serving a small, open-source LLM behind `vLLM`, containerized on a single desktop GPU. 

The target is to have a single-node Kubernetes, with Grafana to monitor and a throughput benchmark.

> Status: containerized OpenAI-compatible endpoint running on an RTX 3060 Ti (8GB). K8s, monitoring, and benchmark are scoped below but not yet built.

## Why vLLM
Mainly because of the OpenAI-compatible server, so the endpoint is `v1/chat/completions` and any OpenAI client works against it.

However, there are two additional advantages:
1. Instead of using `transformers` in Python `vLLM` delivers [PagedAttention](https://www.runpod.io/articles/guides/vllm-pagedattention-continuous-batching) for sovling memory fragmentation and contiguous allocation. It's similar to virtual memory in OS memory-paging which reduces over-reservation that contiguous per-sequence allocation causes.
2. We get continuous batching, which prevents GPU idling. When the GPU gets a set of requests, it statically batches all until they're ready to be operated on, runs them in parallel then forms the next batch. However, sequences finish at varying different lengths of times, which means that most of the GPU may sit idling waiting for one element in a batch to complete its computation. The continuous batching pardigm batches at the iteration level, not the request level which allows the GPU to be more continually saturated. 



## Entry point: vllm serve

Launch with vllm serve MODEL, not `python -m vllm.entrypoints.openai.api_server --model MODEL`. The latter is deprecated in current vLLM (emits a DeprecationWarning, slated for removal)

## The container: what "match the node driver" actually means

The central fact that drove every base-image decision: the container never contains the GPU driver. It carries CUDA userspace libraries; the driver (libcuda.so) is injected from the host at runtime by the NVIDIA Container Toolkit (--gpus all, or the K8s device plugin). So "match the driver" doesn't mean equal versions — it means the container's CUDA toolkit must be ≤ the maximum CUDA the host driver supports. Newer drivers run older toolkits; the reverse fails.

On this box, nvidia-smi reports driver 595.80 / CUDA 13.2 ceiling, so any CUDA ≤ 13.2 in the container is safe. The RTX 3060 Ti is Ampere, sm_86.

A subtlety that makes the base image's CUDA tag nearly cosmetic for a pip-installed stack: the PyPI torch wheel bundles its own CUDA userspace (nvidia-cublas-cu12, nvidia-cudnn-cu12, nvidia-nccl-cu12…) as pip packages, and its `.sos` carry RUNPATH entries pointing at those bundled libs. So cuBLAS-from-the-image is shadowed by cuBLAS-from-pip. The container carries CUDA userspace via pip; the host injects kernel-space via the driver. The base image's CUDA version is almost a red herring, the pip-bundled CUDA version is the thing actually under the driver-compatibility constraint.


# Failures
There was definitely some experimentation around how I got here regarding the `Dockerfile` so I thought to document certain design decision

## Failure 1 - a release-candidate Python

First hand-rolled multi-stage build (CUDA -runtime base, apt install python3.11, venv, pip install vllm) crashed on import:
```
AttributeError: module 'sys' has no attribute 'get_int_max_str_digits'
```

`sys.get_int_max_str_digits` shipped in every stock CPython ≥ `3.11.0`. Its absence meant the interpreter wasn't a released 3.11.x. It was 3.11.0rc1, a from-source release candidate pulled in by forcing python3.11 onto an Ubuntu base whose native Python is a different version.

**Fix**: never hand-pick a Python minor the base distro doesn't own. Use the base's native python3 (a real released build from the distro archive), and pin the base image so it can't drift. If a newer vLLM needs a newer Python, move the base to one whose native python3 satisfies it (e.g. an ubuntu24.04 CUDA tag → 3.12) and don't install a foreign interpreter.

## Failure 2 - a dangling interpreter symlink
```/opt/venv/bin/python3: bad interpreter: No such file or directory```

A venv is not self-contained: `/opt/venv/bin/python3` is a symlink to the interpreter that built the venv. The multi-stage `COPY --from=builder /opt/venv` carried the symlink into a runtime stage that never installed python3, so the target didn't exist and the shebang couldn't resolve.

**Fix (and the rule)***: the runtime stage must install the same python3 the venv points at, and both stages must use the identical pinned base so the interpreter lives at the same path with a matching glibc/libpython ABI.

## Failure 3 - Hardware reality: 8GB, shared with the desktop

`nvidia-smi` shows 7.66 GiB usable (not a clean 8), with the desktop (gnome-shell, Xwayland, apps) already holding ~1.3GB. That leaves ~6GB for serving.

### Model memory math (fp16 = 2 bytes/param):

3B fp16 ≈ 6.2GB weights — leaves nothing for KV cache. OOM.
1.5B fp16 ≈ 3GB.
0.5B fp16 ≈ 1GB.

The path here is to shrink the model until it fits the actual free budget rather than fight the display for memory. Running the box headless (integrated graphics, or stopping the display manager) reclaims the ~1.3GB and is more effective than any single flag.

### Quantization and the kernel-compilation rabbit hole

To fit 3B I first went 4-bit AWQ (~2GB weights), which pulled in a chain of runtime-compilation failures that turned out to be the most instructive part of the project:

- `Failed to find C compiler`: Triton JIT-compiles a C launcher stub and the -runtime base has no gcc. 
- `/usr/local/cuda/bin/nvcc not found`: a CUDA source compile needs the full toolkit, which lives only in -devel. The logs named the real culprit: FlashInfer. vLLM uses it as the attention/sampling backend, and it JIT-compiles its CUDA kernels from .cu source at first use, cross-compiled for sm_86, via nvcc:
```
flashinfer/jit/cpp_ext.py → run_ninja → nvcc ... flashinfer_sampling_binding.cu
/bin/sh: 1: /usr/local/cuda/bin/nvcc: not found   [code=127]
```

The lesson: three separate subsystems (Triton, torch.compile, FlashInfer) compile at runtime, and a CUDA -runtime base can satisfy none of them. Adding tools one at a time just exposes the next missing one, the endpoint is essentially rebuilding -devel by hand.

**Decision**: base on the official `vllm/vllm-openai` image, which ships a matched CUDA/torch/vLLM/FlashInfer toolchain so these compiles succeed rather than being suppressed. This trades a large image for correctness and reproducibility, where working compiled kernels are the headline and image size is a footnote.

## Memory tuning — utilization is a reservation, not a cap

Even at 1.5B, OOM:

- GPU 0 has total capacity of 7.66 GiB of which 201.69 MiB is free.
- this process has 5.73 GiB memory in use.

- 3GB of weights, but the process held 5.73GB, `--gpu-memory-utilization` is a target reservation: vLLM computes (utilization × total) and pre-reserves that whole block for weights + KV cache, then CUDA-graph capture asks for more on top. 0.70 × 7.66 ≈ 5.4GB collided with the shared budget

### Levers, in order of impact:

`--gpu-memory-utilization` 0.55: lower the reservation to fit under what's actually free (desktop is holding ~1.3GB). Lowering, not raising, is the fix.
`--enforce-eager`: disables CUDA-graph capture, removing the private-pool allocations and the capture-time spike that issued the failing request. Costs some throughput; a deliberate memory-vs-throughput trade on an 8GB shared card.
`--max-model-len 1024`: KV-cache reservation scales with context length; capping it frees cache memory.

# Multi-stage, reconsidered

Multi-stage separates build-time bulk from runtime; it cannot shrink runtime bulk. On the official image the bulk (torch + CUDA userspace + vLLM + FlashInfer) is all required at runtime, and FlashInfer compiles at runtime which meant that a stripped runtime base fails exactly as above. Copying the env onto a slim base would mean re-adding the toolchain (back to -devel) plus risking ABI mismatch: not slimming, just reconstructing the official image badly.
## Building
Where same-base multi-stage does earn its place: baking a genuinely separable artifact to move a runtime cost to build time. The builder downloads the weights; the runtime stage copies them in, converting "pull weights on every cold start" into "already in the image."

```dockerfile
FROM vllm/vllm-openai:v0.28.0 AS fetch
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"

FROM vllm/vllm-openai:v0.28.0
COPY --from=fetch /root/.cache/huggingface /root/.cache/huggingface
CMD ["--model","Qwen/Qwen2.5-1.5B-Instruct",\
     "--gpu-memory-utilization","0.55","--max-model-len","2048","--enforce-eager"]
```

The image's ENTRYPOINT is already vllm serve, so CMD supplies args only. The tradeoff is a larger, immutable, air-gap-ready image with instant cold start, versus a small image that pulls at runtime. 

## Running it: 

```bash
docker build -t llm-serving:gpu .
docker run --rm --gpus all -p 8000:8000 --ipc=host llm-serving:gpu
```

Wait for Uvicorn running on http://0.0.0.0:8000 before querying.

## Querying

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"Say hello in one word."}]}'
```

or in Python:

```python
import requests
r = requests.post("http://localhost:8000/v1/chat/completions",
    json={"model":"Qwen/Qwen2.5-1.5B-Instruct",
          "messages":[{"role":"user","content":"Say hello in one word."}]})
r.raise_for_status()
print(r.json()["choices"][0]["message"]["content"])
```

# What I'd do next (roadmap)
- Single-node Kubernetes (k3s/kind): Deployment with a nvidia.com/gpu resource request, readiness/liveness probes on vLLM's /health, a Service. Note: K8s ignores a container's Docker HEALTHCHECK and uses its own probes.
- Weight persistence in K8s: PVC or emptyDir mounted at the HF cache so pod reschedules don't re-download — same problem the weight-baking solves at image level, different mechanism.
- Monitoring: vLLM exports Prometheus metrics natively (TTFT, inter-token latency, throughput, KV-cache utilization, queue depth). Grafana dashboard on those.
- Benchmark: sweep concurrency 1→50, plot throughput vs. latency; the knee where the KV cache saturates and requests queue is the headline result. Discard the first request (JIT/compile warmup) so cold-start cost doesn't pollute latency numbers.
- Production hardening: run as non-root (runAsNonRoot), a tini/--init shim for PID-1 zombie reaping of vLLM's worker subprocesses, autoscaling on queue depth, multi-replica.

### Environment

RTX 3060 Ti (8GB, Ampere sm_86), driver 595.80 (CUDA 13.2 ceiling), vllm/vllm-openai:v0.28.0. Benchmarks, when added, will be taken headless to avoid contention with the desktop sharing the GPU.