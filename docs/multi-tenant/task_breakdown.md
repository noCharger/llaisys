# Multi-Tenant Implementation Task Breakdown

> Based on `architecture-design.md` (M1: Contiguous Memory) and commit `8ba9527`.

## Status Overview
- **Completed**: Control Plane (Tenant/RateLimit), Queue Service, Adaptive Batching.
- **In Progress / Needs Refactor**: Scheduler (Logic mismatch), KV Manager (Architecture mismatch).
- **Pending**: Auth Middleware, Contiguous KV Allocator, Ragged Batch Kernels.

## Task List

| Task ID | Description | Status | Commit | Acceptance Criteria | Test Case | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Control Plane** | | | | | | |
| 1.1 | **Tenant Manager**: Implement CRUD for tenants, API Key management, and quota storage. | ✅ | `8ba9527` | Can create tenants, generate sk-keys, set Quotas. | `test_control_plane.py` | `src/chatbot/backend/app/services/tenant_manager.py` |
| 1.2 | **Rate Limiter**: Implement Token Bucket based rate limiter. | ✅ | `8ba9527` | Reject requests exceeding `requests_per_minute`. | `test_control_plane.py` | `src/chatbot/backend/app/services/rate_limiter.py` |
| 1.3 | **Auth Middleware**: Implement Gateway layer API Key validation & Context injection. | ✅ | `8ba9527` | `Authorization` header properly maps to `X-Tenant-ID`. | `test_auth_middleware.py` | Rate limiter integration complete. |
| **2. Data Plane** | | | | | | |
| 2.1 | **Queue Service**: Implement in-memory PriorityQueue with Lease/Ack mechanism. | ✅ | `8ba9527` | Support `enqueue`, `lease` (w/ timeout), `ack`, `nack`. | `test_queue_service.py` | `src/chatbot/backend/app/services/queue_service.py` |
| 2.2 | **Scheduler Refactor**: Decouple Lease and Exec logic. | ⚠️ | - | Implement "Continuous Batching Loop": Cleanup -> Inject -> Exec. | `test_scheduler.py` | Current `8ba9527` logic is legacy. **High Priority**. |
| 2.3 | **Adaptive Batching**: Implement AIMD dynamic batch sizing. | ✅ | `8ba9527` | `current_batch_size` adjusts based on latency. | `test_scheduler.py` | Implemented in `ClipperScheduler`. |
| 2.4 | **KV-Cache Manager (Contiguous)**: Implement Contiguous Memory Allocator. | ⚠️ | - | Allocate single contiguous tensor per request (M1 requirement). | `test_core/test_kv_cache.py` | Current `8ba9527` implemented Paged KV (`BlockAllocator`). **Need to implement/switch to Contiguous Allocator for M1 compliance.** |
| **3. Execution Engine** | | | | | | |
| 3.1 | **Batch Router**: Convert RequestJob to Batch Input. | ⚠️ | `8ba9527` | Handle Batch inputs correctly. | - | `_process_batch` exists but lacks Ragged Batch handling. |
| 3.2 | **Kernel - RoPE**: Support Ragged Batch (position_ids). | ☐ | - | RoPE accepts `position_ids` tensor. | - | Critical for Continuous Batching. |
| 3.3 | **Kernel - Attention**: Support Variable Length (cu_seqlens). | ☐ | - | Attention accepts `cu_seqlens`. | - | Critical for Continuous Batching. |
| **4. Gateway & API** | | | | | | |
| 4.1 | **Chat API**: Implement `/v1/chat/completions`. | ✅ | `8ba9527` | Standard OpenAI format compliance. | `test_chat_service.py` | `src/chatbot/backend/app/routers/chat.py` |
| 4.2 | **Cache Affinity**: Implement Session ID routing affinity. | ☐ | - | Same Session ID routes to same instance. | - | M2 Milestone. |
