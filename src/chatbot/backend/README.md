# LLAISYS Chatbot Backend

Multi-tenant inference service implementing **Project #4**: continuous batching
with per-request KV cache pooling, packed/varlen attention, and SSE streaming.

## Architecture

```
HTTP request
   │
   ▼
AuthMiddleware (Bearer API key → tenant_id)         (app/middleware/auth.py)
   │
   ▼
RateLimiter (per-tenant token bucket)               (services/rate_limiter.py)
   │
   ▼
Router /v1/chat/completions   ──── stream=true ──►  SSE generator
   │                                                    │
   │                                                    ▼
   ├── ChatService.chat_completion (non-stream)    job.tokens_queue
   │                                                    ▲
   ▼                                                    │
ClipperScheduler (3-phase loop)  ◄─────────── tokens pushed per step
   │
   ├── Cleanup:  release pool blocks for finished/cancelled jobs, ack/nack
   ├── Inject:   lease new jobs, pool.acquire(tenant_id, prefix), set Phase
   └── Execute:  pack all active_requests → forward_batch → push tokens
                          │
                          ▼
                   BatchedQwen2Service        (services/model_service.py)
                          │
                          ▼
                   Qwen2.forward_batch_paged  (Python ctypes wrapper)
                          │
                          ▼
                   llaisysQwen2ModelForwardBatchPool   (C++)
                          │
                          ▼
                   Embedding → Layers (RMS, QKV linear, RoPE, varlen attn,
                                        O proj, residual, MLP)
                          │
                          ▼
                   lm_head per request → batched random_sample
```

### Tenant isolation

`tenant_id` flows from `request.state.tenant_id` (set by `AuthMiddleware` after
validating the Bearer API key) into every `RequestJob`. The scheduler hands
that string to `KVPoolService.acquire(tenant_id, ...)`, which is the **only**
path through which the KV cache pool is touched.

The C++ pool enforces:

* **Same-tenant prefix matching + LRU**: a tenant resending the prefix of an
  earlier turn reuses the same block; only the suffix is prefilled.
* **Cross-tenant wipe-on-reuse**: when LRU eviction crosses a tenant boundary,
  the K/V tensors are zero-filled before the new tenant takes ownership. The
  wipe runs **outside** the pool mutex so other threads' acquires aren't
  starved by long CUDA memsets.

The auth middleware **rejects unauthenticated requests** with 401. The
`x-tenant-id` header is no longer accepted as a fallback (would be a tenant
spoofing bypass).

### KV Cache Pool

Configured in `dependencies.py`:

```python
kv_pool_service = KVPoolService(
    pool=None,                          # stub for tests; replace at startup
    n_blocks=32,
    max_len_per_block=8192,
)
```

Real-mode wiring in production startup (e.g., `app/main.py`):

```python
from llaisys.models import Qwen2, PagedKVCache
from .services.tokenizer import build_tokenizer
from .services.model_service import BatchedQwen2Service

qwen2 = Qwen2(model_path=..., device=DeviceType.NVIDIA, dtype=DataType.F16)
real_pool = PagedKVCache(qwen2, n_pages=512, page_size=16, max_pages_per_request=512)
kv_pool_service = KVPoolService(
    pool=real_pool, n_blocks=64, max_len_per_block=8192,
)
model_service = BatchedQwen2Service(qwen2, kv_pool_service)
scheduler = ClipperScheduler(model_service, queue_service, kv_pool_service,
                             eos_token=qwen2.end_token)
```

### Continuous batching: 3-phase loop

`services/scheduler.py` runs:

```
while running:
    mark_cancelled_jobs_finished()      # detect future.cancel()
    cleanup_finished()                  # release blocks, ack/nack
    inject_new_requests()               # pool.acquire, prefix matching
    if no active jobs: wait on Event    # idle wakeup, no busy spin
    execute_batch()                     # pack & forward_batch → stream tokens
    update_batch_size(latency)          # AIMD vs slo_ms
```

Each iteration produces **one token per active request** (mixed prefill +
decode). When a request hits EOS or `max_tokens`, it is marked finished and
its KV block is released in the next Cleanup phase.

### Streaming

`/v1/chat/completions` with `stream: true` returns SSE chunks. The router
holds a per-job `asyncio.Queue` that the scheduler pushes tokens into; an
`async generator` yields OpenAI-compatible delta chunks until the `DONE`
sentinel arrives. Token decoding goes through the configured `Tokenizer`
abstraction (`HFTokenizer` or `CharTokenizer`) and uses incremental
`decode_step(accumulated_ids, new_id)` so BPE/SentencePiece subword
boundaries don't fragment in the stream.

### Tokenizer

`services/tokenizer.py` exposes a `build_tokenizer(path)` factory:

* `LLAISYS_TOKENIZER_PATH` env var → loads HuggingFace `AutoTokenizer` via
  `transformers`.
* Otherwise → `CharTokenizer` placeholder (each char → its codepoint), used by
  the unit/integration tests so they don't need a real model on disk.

When the real tokenizer is loaded, the scheduler is auto-configured with
`scheduler.set_eos_token(tokenizer.eos_token_id)` so generation auto-terminates.

## Files

| Module | Role |
|---|---|
| `app/main.py` | FastAPI app, CORS, AuthMiddleware, router wiring |
| `app/dependencies.py` | DI container (single instances of services) |
| `app/middleware/auth.py` | Bearer API key → `request.state.tenant_id` |
| `app/routers/chat.py` | `/v1/chat/completions` + SSE streaming |
| `app/routers/admin.py` | Tenant CRUD + API key generation |
| `app/services/scheduler.py` | 3-phase continuous-batching scheduler |
| `app/services/kv_pool_service.py` | Tenant-scoped wrapper around KV pool |
| `app/services/model_service.py` | Stub + `BatchedQwen2Service` |
| `app/services/tokenizer.py` | HF / placeholder tokenizer abstraction |
| `app/services/queue_service.py` | In-memory priority queue with lease/ack |
| `app/services/rate_limiter.py` | Per-tenant token bucket |
| `app/services/tenant_manager.py` | Tenant + API key management |
| `app/services/chat_service.py` | Non-streaming chat completion |
| `app/services/context_manager.py` | Jinja2 system-prompt rendering |
| `app/services/storage.py` | Storage backend ABC + in-memory mock |

## Tests

```bash
PYTHONPATH=src python -m pytest test/chatbot/
```

Key coverage:

* `test_continuous_batching.py` — concurrent SSE streams, tenant isolation,
  block release on finish.
* `test_scheduler.py` — AIMD logic, delayed batching.
* `test_queue_integration.py` — queue lease/ack with the scheduler.
* `test_auth_middleware.py` — Bearer auth, rejection paths.
* `test_integration_flow.py` — end-to-end via TestClient with API key.

## Known limitations

* The real-mode wiring (`BatchedQwen2Service`) requires a Qwen2 model on
  disk; the default `dependencies.py` configuration is stub mode for CI.
* Future cancellation is observed only at the next scheduler tick; in-flight
  forward calls cannot be aborted mid-call.
* The scheduler currently uses a single `asyncio.Event` for idle wakeup; if
  multiple submitters race they'll all converge after one tick.
* CUDA varlen kernel is per-request `cublasGemmStridedBatchedEx` over heads
  (no cross-request fusion). Future work: `cublasGemmGroupedBatchedEx` or a
  custom paged-attention kernel.

## Running locally (stub mode)

```bash
PYTHONPATH=src python -m uvicorn chatbot.backend.app.main:app --reload
```

Create a tenant + API key via the admin route, then:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":false}'
```
