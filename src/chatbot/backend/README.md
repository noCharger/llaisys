# LLAISYS Chatbot Backend

Multi-tenant inference service implementing **Project #4**: continuous
batching over a paged-attention KV cache, with chunked prefill, preemption
recovery, and SSE streaming.

## Architecture

```
HTTP request
   │
   ▼
AuthMiddleware                                       app/middleware/auth.py
   Bearer API key resolves to tenant_id
   │
   ▼
RateLimiter                                          services/rate_limiter.py
   Per-tenant token bucket
   │
   ▼
Router /v1/chat/completions  ── stream=true ───►  SSE generator
   │                                                   │
   │                                                   ▼
   ├── ChatService.chat_completion non-stream    job.tokens_queue
   │                                                   ▲
   ▼                                                   │
ClipperScheduler 3-phase loop  ◄────────── tokens pushed per step
   │
   ├── Cleanup  release pool blocks, commit prefix history, ack/nack queue
   ├── Inject   acquire pool blocks, prefix-match, preempt on exhaustion
   └── Execute  pack under token_budget, mix prefill chunks with decode
                          │
                          ▼
                   BatchedQwen2Service               services/model_service.py
                          │
                          ▼
                   Qwen2.forward_batch_paged         Python ctypes wrapper
                          │
                          ▼
                   llaisysQwen2ModelForwardBatchPaged    C++/CUDA
                          │
                          ▼
                   Embedding, layers with paged self-attention,
                   per-row batched random_sample
```

### KV cache pool: paged attention

The C++ pool in `src/llaisys/qwen2_paged_pool.cc` manages a fixed set of
physical pages. Each page holds `page_size` tokens of K and V for all
layers. A request owns a block table mapping its logical positions to
physical page ids. `Append` allocates pages, `Commit` seals fully-filled
pages and registers their chain hash, `Release` returns the block to the
free pool while keeping sealed pages reusable through the prefix index.

The prefix index is **content-addressed**, not partitioned by tenant. A
chain hash maps to the page that holds those exact prefix tokens regardless
of original owner. Two tenants submitting the same prompt land on the same
physical pages. This is safe by determinism: identical input tokens produce
identical KV under a deterministic forward, so cross-tenant sharing is pure
memoization and never leaks data either tenant did not already have.

Within a tenant, shared prefix pages fork via copy-on-write when a session
writes past the shared region. Cross-tenant page takeover zero-wipes the
page contents before reuse, so unrelated tenants never observe each other's
KV.

### Tenant isolation

`tenant_id` flows from `request.state.tenant_id` set by `AuthMiddleware`
into every `RequestJob`, and the scheduler hands it to every
`KVPoolService` call. Isolation is enforced at three layers:

* **Auth**: `AuthMiddleware` rejects unauthenticated requests with 401. The
  `x-tenant-id` header is not accepted as a fallback because it would
  permit tenant spoofing.
* **Quota**: each tenant gets a `Quota.kv_pages_*` triple of
  `reservation_floor`, `max_pages`, and `burst_pages`. The C pool refuses
  allocations beyond `max_pages` and protects `reservation_floor` from
  cross-tenant eviction.
* **Routing**: per-job `tokens_queue` and `future` carry results back to
  the originating request only.

### Per-tenant quota

Configured via `TenantManager.Quota`:

```python
Quota(
    kv_pages_reservation_floor=8,   # never evictable for other tenants
    kv_pages_max=64,                # hard ceiling per tenant
    kv_pages_burst=16,              # opportunistic above floor
)
```

`KVPoolService` pushes these into the C pool on first acquire for each
tenant. A value of 0 means use pool defaults.

### Continuous batching: 3-phase loop

`services/scheduler.py` runs:

```
while running:
    mark_cancelled_jobs_finished()      # detect future.cancel()
    cleanup_finished()                  # commit history, release blocks, ack
    inject_new_requests()               # acquire, prefix-match, preempt
    if no active jobs: wait on Event    # idle wakeup, no busy spin
    execute_batch()                     # pack under token_budget, forward
    update_batch_size(latency)          # AIMD against slo_ms
```

**Chunked prefill**. `token_budget` caps tokens per Execute step. Decode
jobs claim 1 token each; remaining budget is split across prefill jobs in
FIFO order. A long prompt is fed in chunks across consecutive ticks, so a
single 8k prompt no longer pins the GPU for a giant step. Mid-prefill
sampled tokens are discarded; only the post-prefill sample becomes the
first output token.

**Mixed prefill and decode in one step**. Sarathi-Serve style combined
batching: a prefill chunk and active decodes pack into the same varlen
forward through `cu_seqlens_q`. This keeps short decodes from waiting
behind a long prefill.

**Preemption with prefix recovery**. When the pool is exhausted on
acquire, the scheduler evicts the oldest active job, commits its full
input plus output history, and re-queues it. On resume the next acquire
recovers the cached prefix through the global chain index, so re-prefill
skips the cached portion. `max_preemptions_per_job` bounds thrash.

### Streaming

`/v1/chat/completions` with `stream: true` returns SSE chunks. The router
holds a per-job `asyncio.Queue`; the scheduler pushes ints into it as they
are sampled. The async generator yields OpenAI-compatible delta chunks
until the `DONE` sentinel. Decoding goes through the `Tokenizer`
abstraction `HFTokenizer` or `CharTokenizer` using incremental
`decode_step(accumulated_ids, new_id)` so BPE merges do not fragment in
the stream.

### Tokenizer

`services/tokenizer.py` exposes a `build_tokenizer(path)` factory:

* `LLAISYS_TOKENIZER_PATH` env var loads a HuggingFace `AutoTokenizer`.
* Otherwise the placeholder `CharTokenizer` maps each char to its
  codepoint, used by tests so they do not need a real model on disk.

When the real tokenizer is loaded, the scheduler is auto-configured with
`scheduler.set_eos_token(tokenizer.eos_token_id)`.

### Wiring

Default test wiring in `dependencies.py` runs in stub mode:

```python
kv_pool_service = KVPoolService(
    pool=None, n_blocks=32, max_len_per_block=8192,
    tenant_manager=tenant_manager,
)
scheduler = ClipperScheduler(model_service, queue_service, kv_pool_service)
```

Production wiring loads a real Qwen2 model and a paged KV cache:

```python
from llaisys.models import Qwen2
from llaisys.models.qwen2 import PagedKVCache
from .services.model_service import BatchedQwen2Service
from .services.tokenizer import build_tokenizer

qwen2 = Qwen2(model_path=..., device=DeviceType.NVIDIA, dtype=DataType.F16)
paged_pool = PagedKVCache(
    qwen2, n_pages=512, page_size=16, max_pages_per_request=512,
)
kv_pool_service = KVPoolService(
    paged_pool=paged_pool, tenant_manager=tenant_manager,
)
model_service = BatchedQwen2Service(qwen2, kv_pool_service)
scheduler = ClipperScheduler(
    model_service, queue_service, kv_pool_service,
    eos_token=qwen2.end_token, token_budget=4096,
)
tokenizer = build_tokenizer(tokenizer_path=...)
scheduler.set_eos_token(tokenizer.eos_token_id)
```

## Files

| Module | Role |
|---|---|
| `app/main.py` | FastAPI app, CORS, AuthMiddleware, router wiring |
| `app/dependencies.py` | DI container, single instances per service |
| `app/middleware/auth.py` | Bearer API key resolves to `request.state.tenant_id` |
| `app/routers/chat.py` | `/v1/chat/completions` plus SSE streaming |
| `app/routers/admin.py` | Tenant CRUD, API key generation |
| `app/services/scheduler.py` | 3-phase continuous-batching scheduler |
| `app/services/kv_pool_service.py` | Tenant-scoped wrapper around KV pool |
| `app/services/model_service.py` | Stub plus `BatchedQwen2Service` |
| `app/services/tokenizer.py` | HF and placeholder tokenizer abstraction |
| `app/services/queue_service.py` | In-memory priority queue with lease/ack |
| `app/services/rate_limiter.py` | Per-tenant token bucket |
| `app/services/tenant_manager.py` | Tenants, API keys, quotas |
| `app/services/chat_service.py` | Non-streaming chat completion |
| `app/services/context_manager.py` | Jinja2 system-prompt rendering |
| `app/services/storage.py` | Storage backend ABC plus in-memory mock |

## Tests

```bash
PYTHONPATH=src python -m pytest test/chatbot/
```

Key coverage:

* `test_continuous_batching.py` concurrent SSE streams, tenant isolation,
  block release on finish.
* `test_chunked_prefill_preempt.py` chunked prefill under token_budget,
  hash-chain prefix match, cross-tenant prefix sharing, preemption with
  prefix recovery, preempt cap.
* `test_paged_quota.py` per-tenant quota propagation from TenantManager
  into the C pool.
* `test_scheduler.py` AIMD logic, delayed batching.
* `test_queue_integration.py` queue lease/ack with the scheduler.
* `test_auth_middleware.py` Bearer auth, rejection paths.
* `test_integration_flow.py` end-to-end via TestClient with API key.

Native pool tests live one level up at `test/test_paged_pool.py`,
`test/test_paged_cow.py`, `test/test_paged_scatter.py`,
`test/test_forward_batch_paged.py`.

## Known limitations

* Real-mode wiring requires a Qwen2 model on disk. Default
  `dependencies.py` runs stub mode for CI.
* Future cancellation is observed only at the next scheduler tick;
  in-flight forward calls cannot be aborted mid-call.
* Queue-mode preemption is not supported; the in-memory priority queue
  cannot push a message to the front. Preemption is enabled only in
  internal-queue mode.
* Multiple submitters share one `asyncio.Event` for idle wakeup; they all
  converge after one tick.

## Running locally in stub mode

```bash
PYTHONPATH=src python -m uvicorn chatbot.backend.app.main:app --reload
```

Create a tenant and API key via the admin route, then:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":false}'
```
