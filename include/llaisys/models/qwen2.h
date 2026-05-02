#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;
        llaisysTensor_t *attn_norm_w;
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w;
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysQwen2Model;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Session *llaisysQwen2ModelCreateSession(struct LlaisysQwen2Model * model);
    __export void llaisysQwen2ModelDestroySession(struct LlaisysQwen2Session * session);
    __export void llaisysQwen2ModelRewindSession(struct LlaisysQwen2Session * session, size_t len);
    __export int64_t llaisysQwen2ModelForward(struct LlaisysQwen2Session * session, int64_t * token_ids, size_t ntoken, float temp, float top_p, int top_k);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken, float temp, float top_p, int top_k);

    // ----- Paged KV cache pool -----
    // Shared physical-page pool, per-request block tables, ref-counted CoW,
    // per-tenant quotas. See src/llaisys/qwen2_paged_pool.cc for details.
    struct LlaisysQwen2PagedPool;

    __export struct LlaisysQwen2PagedPool *llaisysQwen2PagedPoolCreate(
        struct LlaisysQwen2Model *model,
        size_t n_pages,
        size_t page_size,
        size_t max_pages_per_request);
    __export void llaisysQwen2PagedPoolDestroy(struct LlaisysQwen2PagedPool *pool);

    __export void llaisysQwen2PagedPoolSetTenantQuota(
        struct LlaisysQwen2PagedPool *pool, uint64_t tenant_id,
        size_t reservation_floor, size_t max_pages, size_t burst_pages);

    // Returns block_id, or -1 if pool/quota exhausted (caller may preempt).
    // Matched prefix pages are ref-counted; the unmatched tail is allocated
    // by Append, not here.
    __export int32_t llaisysQwen2PagedPoolAcquire(
        struct LlaisysQwen2PagedPool *pool,
        uint64_t tenant_id,
        const int64_t *prefix_tokens, size_t nprefix,
        size_t *matched_prefix_len);

    __export void llaisysQwen2PagedPoolRelease(
        struct LlaisysQwen2PagedPool *pool, int32_t block_id);

    // Allocates pages as needed. Fills `out_slot_mapping[i] = page_id<<16 |
    // offset`. Returns 0 on success, -1 if quota / pool exhausted.
    __export int32_t llaisysQwen2PagedPoolAppend(
        struct LlaisysQwen2PagedPool *pool, int32_t block_id,
        size_t n_new_tokens,
        int32_t *out_slot_mapping);

    // Install chain hashes for newly-completed pages. `tokens` is the full
    // token history up to `new_pos`.
    __export void llaisysQwen2PagedPoolCommit(
        struct LlaisysQwen2PagedPool *pool, int32_t block_id,
        size_t new_pos,
        const int64_t *tokens, size_t ntokens);

    __export size_t llaisysQwen2PagedPoolBlockPos(
        struct LlaisysQwen2PagedPool *pool, int32_t block_id);
    __export size_t llaisysQwen2PagedPoolPageTable(
        struct LlaisysQwen2PagedPool *pool, int32_t block_id,
        int32_t *out_pages);

    // PageK/PageV return new tensor views; caller must tensorDestroy.
    // BigK/BigV are pool-owned; do NOT destroy.
    __export llaisysTensor_t llaisysQwen2PagedPoolPageK(
        struct LlaisysQwen2PagedPool *pool, int32_t page_id, size_t layer);
    __export llaisysTensor_t llaisysQwen2PagedPoolPageV(
        struct LlaisysQwen2PagedPool *pool, int32_t page_id, size_t layer);
    __export llaisysTensor_t llaisysQwen2PagedPoolBigK(
        struct LlaisysQwen2PagedPool *pool, size_t layer);
    __export llaisysTensor_t llaisysQwen2PagedPoolBigV(
        struct LlaisysQwen2PagedPool *pool, size_t layer);

    __export size_t llaisysQwen2PagedPoolNumPages(struct LlaisysQwen2PagedPool *pool);
    __export size_t llaisysQwen2PagedPoolPageSize(struct LlaisysQwen2PagedPool *pool);
    __export size_t llaisysQwen2PagedPoolTenantPagesUsed(
        struct LlaisysQwen2PagedPool *pool, uint64_t tenant_id);
    __export size_t llaisysQwen2PagedPoolGlobalPagesFree(
        struct LlaisysQwen2PagedPool *pool);

    // Scatter packed K/V [n_tokens, nkvh, dh] into pool pages via
    // slot_mapping[i] = page_id<<16 | offset.
    __export int32_t llaisysQwen2PagedPoolScatterKV(
        struct LlaisysQwen2PagedPool *pool,
        size_t layer,
        llaisysTensor_t k_new,
        llaisysTensor_t v_new,
        const int32_t *slot_mapping,
        size_t n_tokens);

    // Batched forward over packed tokens. Caller must have already Appended
    // pages for every block (slot_mapping is the Append output) and will
    // separately Commit the token history after this returns.
    // Returns 0 on success, negative on error; zero-fills out_next_tokens on error.
    __export int32_t llaisysQwen2ModelForwardBatchPaged(
        struct LlaisysQwen2Model *model,
        struct LlaisysQwen2PagedPool *pool,
        const int64_t *packed_tokens,
        const int32_t *cu_seqlens_q,
        const int32_t *block_ids,
        const int32_t *slot_mapping,
        const float *temps,
        const float *top_ps,
        const int *top_ks,
        int32_t batch,
        int64_t *out_next_tokens
    );
}
#endif // LLAISYS_MODELS_QWEN2_H
