# LLAISYS — Let's Learn AI SYStem

<p align="center">
  <a href="https://github.com/noCharger/llaisys/actions/workflows/build.yaml">
    <img src="https://github.com/noCharger/llaisys/actions/workflows/build.yaml/badge.svg" alt="Build">
  </a>
  <img src="https://img.shields.io/badge/CUDA-12.x-76b900?logo=nvidia" alt="CUDA 12">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue?logo=c%2B%2B" alt="C++17">
  <img src="https://img.shields.io/badge/Python-%E2%89%A53.9-3776ab?logo=python&logoColor=white" alt="Python ≥3.9">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
</p>

<p align="center">
  <b>A production-grade LLM inference engine built from scratch —<br>
  custom CUDA kernels, continuous batching, paged KV cache, and multi-tenant serving.</b>
</p>

<p align="center">
  <a href="README.md">English</a> ｜
  <a href="README_ZN.md">中文</a> ｜
  <a href="docs/assignments.md">Assignment Spec</a>
</p>

---

## What is LLAISYS?

LLAISYS bridges the gap between theory and production. Starting from a blank C++ template, it grows into a full-stack LLM serving system: a hand-rolled tensor library, custom CUDA kernels that beat PyTorch Eager on key transformer operators, an OpenAI-compatible REST server with streaming and session management, and a multi-tenant continuous-batching engine with a paged KV-cache pool.

**Stack:** C++17 shared library (`libllaisys`) → Python ctypes layer → FastAPI server → React 18 + Vite chat UI  
**Model:** Qwen2 / DeepSeek-R1-Distill-Qwen-1.5B

---

## System Architecture

```
┌─────────────────────────────────────────────────┐
│  React 18 + Vite  (port 5173)                   │
│  Real-time SSE streaming · Thought Process UI   │
└──────────────────┬──────────────────────────────┘
                   │ REST / SSE
┌──────────────────▼──────────────────────────────┐
│  FastAPI Server  (port 6008 / 6009 HTTPS)       │
│  Session Manager · Context Manager · Auth       │
│  Prometheus metrics · OpenTelemetry tracing     │
└──────────────────┬──────────────────────────────┘
                   │ ctypes / FFI
┌──────────────────▼──────────────────────────────┐
│  libllaisys  (C++17 shared library)             │
│  Tensor · Ops · KV-Cache · Paged KV Pool        │
│  Qwen2 model forward pass                       │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │  CPU                │  NVIDIA GPU (CUDA 12)
        └─────────────────────┘
```

---

## What Was Built

### Milestone 1 — Tensor & CPU Operators

A custom tensor abstraction with zero-copy `view`, `permute`, and `slice`, backed by a reference-counted storage block shared across devices. Seven CPU transformer operators implemented from scratch in C++:

`argmax` · `embedding` · `linear` · `rms_norm` · `rope` · `self_attention` · `swiglu`

All operators support FP32, FP16, and BF16. KV-cache with prefix matching enables multi-turn dialogue without recomputing shared context.

### Milestone 2 — CUDA Acceleration

A clean NVIDIA runtime abstraction mirrors the CPU API surface — the model layer sees one unified interface. Custom CUDA kernels push operator performance well past PyTorch Eager:

- **float4 vectorized loads/stores** in `add`, `linear` (bias), and `rms_norm` for peak memory bandwidth
- **Fused kernels**: rms_norm (Square→Mean→Normalize in one pass), swiglu (Gate→Silu→Mul)
- **Static `thread_local` cuBLAS handle** eliminates ~300 µs creation overhead per `linear` call
- **Shared memory reductions** in `rms_norm` and `argmax`
- Random sampling on GPU: CUB `DeviceRadixSort` + `DeviceScan::InclusiveSum` for Top-K/Top-P

### Milestone 3 — AI Chatbot Server

A full-featured chat service on top of the inference engine:

- **Random sampling** (Temperature, Top-K, Top-P) replaces deterministic argmax
- **OpenAI-compatible API** — `POST /v1/chat/completions` with SSE streaming
- **HTTPS** via Uvicorn + SSL; security headers (CSP, X-Frame-Options) enforced by middleware
- **Thought Process visualization** — parses `<think>` tags and renders model reasoning in the UI
- **Session management** — LRU eviction, `in_use` flag prevents concurrent session corruption
- **Observability** — Prometheus metrics (`/metrics`), OpenTelemetry tracing, Grafana dashboard

### Milestone 4 — Multi-Tenant Serving

A continuous-batching engine that handles many users simultaneously:

- **Iteration-level scheduler** — each forward step draws a fresh variable-length batch from the request pool, maximizing GPU utilization
- **Paged KV-cache pool** — reference-counted copy-on-write blocks with per-request block tables, eliminating memory fragmentation
- **Global prefix index** — cross-tenant KV reuse; common system prompts are computed once and shared
- **Auth layer** — API key hashing, Admin UI, per-tenant quotas
- **Batch operators** — `self_attention_varlen`, `self_attention_paged` for heterogeneous-length batches

---

## Performance

Benchmarked against PyTorch 2.x on NVIDIA GPU (CUDA 12). LLAISYS is especially fast in latency-sensitive, small-batch scenarios where Python dispatch overhead dominates.

| Operator  | Shape       | Precision | LLAISYS  | vs PyTorch Eager | vs Torch Compile |
|-----------|-------------|-----------|----------|-----------------|-----------------|
| RoPE      | (2, 1, 4)   | FP16      | 0.006 ms | **13.82×**      | 3.38×           |
| RMS Norm  | (1, 4)      | FP32      | 0.007 ms | **4.26×**       | 3.07×           |
| SwiGLU    | (512, 4096) | BF16      | 0.013 ms | **4.27×**       | 3.61×           |
| Attention | q=2, kv=2   | FP32      | 0.014 ms | **6.52×**       | 4.40×           |
| Linear    | (2, 3)      | FP32      | 0.011 ms | 1.03×           | 2.85×           |

> For large matrix ops (e.g. Linear 512×4096) LLAISYS matches cuBLAS peak throughput. Speedups are largest for normalization and element-wise ops where the C++ runtime overhead is minimal compared to PyTorch's Python dispatch path.

---

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 12.x
- Python ≥ 3.9
- [xmake](https://xmake.io/) build tool

### Build

```bash
# Compile C++ backend with CUDA support
xmake f --nv-gpu=y -cv
xmake && xmake install

# Install Python package
pip install ./python/
```

CPU-only build (no GPU required):

```bash
xmake && xmake install
pip install ./python/
```

### Run the Chat Server

```bash
export MODEL_PATH="/path/to/DeepSeek-R1-Distill-Qwen-1.5B"
python scripts/start.py --device nvidia --max-steps 50000 --model $MODEL_PATH
```

```
[INFO] Starting Backend Server...
[INFO] Successfully connected to http://localhost:6008/health
[INFO] ALL SERVICES STARTED SUCCESSFULLY
[INFO] Frontend URL: http://localhost:5173
[INFO] Backend  URL: http://localhost:6008
```

Open `http://localhost:5173` to start chatting.

### Run Benchmarks

```bash
# Operator performance vs PyTorch
python test/test_ops.py --device nvidia --profile

# End-to-end inference correctness
python test/test_infer.py --model $MODEL_PATH --device nvidia --test
```

### API Quick Call

```bash
curl -X POST http://localhost:6008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "stream": true
  }'
```

---

## Learning Path

The assignments and projects that built this system are documented in [docs/assignments.md](docs/assignments.md) ([中文](docs/assignments_zh.md)).

| Step | Topic | What you implement |
|------|-------|--------------------|
| Assignment 0 | Getting Started | Build system, test harness |
| Assignment 1 | Tensor | `view`, `permute`, `slice`, device memory |
| Assignment 2 | CPU Operators | All 7 transformer ops in C++ |
| Assignment 3 | LLM Inference | Qwen2 forward pass + KV-cache |
| Project 1 | CPU Optimization | SIMD / OpenMP / BLAS tuning |
| Project 2 | CUDA | Custom kernels, cuBLAS, CUDA runtime |
| Project 3 | Chatbot | Server, UI, random sampling, sessions |
| Project 4 | Multi-tenant | Continuous batching, paged KV pool |
| Project 5 | Distributed | Tensor parallelism with NCCL |
| Project 6 | New Models | Extend beyond Qwen2 |

---

## License

MIT © [InfiniTensor](https://github.com/InfiniTensor) — see [LICENSE](LICENSE).  
Technical Report authored by **no_charger** (March 2026).
