# PoC Plan: triton-sigmoid

## Project Classification
- **Type:** infrastructure (GPU kernel library)
- **Key Technologies:** Python, PyTorch (≥2.6), OpenAI Triton (≥3.2), CUDA, NumPy, Pandas
- **ODH Relevance:** This library provides high-performance fused sigmoid attention kernels for NVIDIA GPUs. It is directly relevant to AI/ML workloads on Open Data Hub — it can accelerate attention computation in transformer models deployed or trained on OpenShift AI clusters. Validating that the library works inside a containerized GPU environment on OpenShift proves it can be used as a dependency in ODH model training and serving pipelines.

## PoC Objectives
What we want to prove:
1. The `triton-sigmoid` library can be installed and imported inside a containerized environment with GPU support
2. The **dense** sigmoid attention kernel executes correctly on GPU, producing tensors of the expected shape (both causal and non-causal modes)
3. The **padded** sigmoid attention kernel handles variable-length sequences correctly on GPU
4. The library is compatible with `torch.compile` for graph-mode optimization
5. The container can run on an OpenShift/Kubernetes cluster with NVIDIA GPU resources

## Infrastructure Requirements
- **Inference Server:** none — this is a library, not a server
- **Vector Database:** none
- **Embedding Model:** none
- **GPU Required:** yes — NVIDIA Ampere architecture (A100, H100) or newer. The Triton kernels are compiled for and execute on NVIDIA GPUs; there is no CPU fallback.
- **Persistent Storage:** none
- **Resource Profile:** gpu (8Gi RAM, 4 CPU, 1 NVIDIA GPU)
- **Sidecar Containers:** none

## Test Scenarios

### Scenario 1: Import Check
- **Description:** Verify the `triton_sigmoid` package imports successfully and its public API (`sigmoid_attention`, `sigmoid_attention_padded`) is available
- **Type:** cli
- **Input:** `python -c "from triton_sigmoid import sigmoid_attention, sigmoid_attention_padded; print('Import OK')"`
- **Expected:** Job exits 0, output contains "Import OK"
- **Timeout:** 60 seconds

### Scenario 2: Dense Attention Forward Pass
- **Description:** Execute a dense sigmoid attention forward pass with batch=2, seq_len=1024, n_heads=8, head_dim=64 in float16 on GPU (non-causal mode)
- **Type:** cli
- **Input:** Python script creating random Q, K, V tensors and calling `sigmoid_attention(q, k, v, is_causal=False)`
- **Expected:** Job exits 0, output tensor has shape (2, 1024, 8, 64) and float16 dtype
- **Timeout:** 120 seconds

### Scenario 3: Dense Attention with Causal Masking
- **Description:** Execute dense sigmoid attention with `is_causal=True` to verify causal masking works
- **Type:** cli
- **Input:** Python script calling `sigmoid_attention(q, k, v, is_causal=True)` with smaller tensors
- **Expected:** Job exits 0, correct output shape
- **Timeout:** 120 seconds

### Scenario 4: Padded Attention Forward Pass
- **Description:** Execute padded sigmoid attention with variable-length sequences (seq_lens_k=[800, 950], seq_lens_q=[800, 950]) to verify padding mask handling
- **Type:** cli
- **Input:** Python script calling `sigmoid_attention_padded(q, k, v, seq_lens_k=..., seq_lens_q=...)`
- **Expected:** Job exits 0, output tensor has shape (2, 1024, 8, 64)
- **Timeout:** 120 seconds

### Scenario 5: torch.compile Compatibility
- **Description:** Verify that `torch.compile(sigmoid_attention_padded)` works, proving the Triton kernels are compatible with PyTorch's graph compiler
- **Type:** cli
- **Input:** Python script wrapping `sigmoid_attention_padded` with `torch.compile` and executing with causal masking
- **Expected:** Job exits 0, compiled function produces correct output shape
- **Timeout:** 180 seconds (torch.compile compilation can be slow on first run)

## Dockerfile Considerations
This is a **GPU kernel library**, not a server. The Dockerfile must:

- **Base image:** Use an NVIDIA CUDA base image with Python 3.11 or 3.12 (e.g., `nvidia/cuda:12.8.0-devel-ubi9` or a PyTorch NGC container). The `devel` variant is needed because Triton compiles PTX/SASS at runtime.
- **Install PyTorch with CUDA support:** Install `torch>=2.6.0` from the PyTorch CUDA 12.8 wheel index (`https://download.pytorch.org/whl/cu128`).
- **Install triton>=3.2.0:** This is a build/runtime dependency for the GPU kernels.
- **Install the library:** `pip install .` from the repo root using `pyproject.toml`.
- **ENTRYPOINT:** Set to `python` so test Jobs can pass `-c "..."` as arguments.
- **CMD:** Default to `--version` or `-c "import triton_sigmoid; print('triton-sigmoid ready')"` so the container produces useful output when run without arguments.
- **Do NOT add EXPOSE** — there is no port to expose. This is a library, not a server.
- **Do NOT run a server process** — the container should exit after executing whatever command is passed to it.

## Deployment Considerations
- **Deployment model:** `Job` — This is a library. It does NOT run as a long-lived server. Each test scenario creates a Kubernetes Job that runs a Python command, checks exit code + logs, and completes. **Do NOT deploy as a Deployment** — it will CrashLoopBackOff endlessly because there's no process to keep running.
- **Do NOT create a Service** — there is no port to expose. The library does not listen on any port.
- **GPU resources:** Each Job must request `nvidia.com/gpu: 1` in its resource limits. The cluster must have NVIDIA GPU Operator installed and nodes with Ampere (A100) or newer GPUs.
- **NVIDIA runtime:** The Pod spec needs `runtimeClassName: nvidia` or the equivalent toleration/nodeSelector for GPU nodes.
- **Test via `kubectl run --rm`** or by creating Jobs: Each test scenario runs a `python -c "..."` command inside the container. Success is determined by Job exit code (0 = pass) and log output (must contain expected success strings).
- **No PVC needed** — all test data is generated in-memory at runtime.
- **Timeout consideration:** The first invocation of Triton kernels involves JIT compilation, which can take 30-60 seconds. Set generous timeouts (120-180s) for GPU test scenarios.