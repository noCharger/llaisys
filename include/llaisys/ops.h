#ifndef LLAISYS_OPS_H
#define LLAISYS_OPS_H

#include "tensor.h"

__C {
    __export void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b);
    __export void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals);
    __export void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight);
    __export void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias);
    __export void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in);
    __export void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps);
    __export void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta);
    __export void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale);
    __export void llaisysSelfAttentionVarlen(
        llaisysTensor_t attn_val,
        llaisysTensor_t q,
        llaisysTensor_t *k_blocks,
        llaisysTensor_t *v_blocks,
        const int32_t *cu_seqlens_q,
        int32_t batch,
        float scale);
    __export void llaisysSelfAttentionPaged(
        llaisysTensor_t attn_val,
        llaisysTensor_t q,
        llaisysTensor_t big_k,                  // pool's [n_pages*page_size, nkvh, dh] for one layer
        llaisysTensor_t big_v,
        const int32_t *block_tables,            // flat: npages_r entries per request, concatenated
        const int32_t *block_table_lens,        // [batch]
        const int32_t *cu_seqlens_q,            // [batch+1]
        const int32_t *kv_lens,                 // [batch]
        int32_t batch,
        int32_t page_size,
        float scale);
    __export void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up);
    __export void llaisysRandomSample(llaisysTensor_t out_token, llaisysTensor_t logits, float temp, float top_p, int top_k);
    __export void llaisysRandomSampleBatch(
        llaisysTensor_t out_tokens,
        llaisysTensor_t logits,
        const float *temps,
        const float *top_ps,
        const int *top_ks);
    __export void llaisysArgmaxBatch(
        llaisysTensor_t out_indices,
        llaisysTensor_t logits);
}

#endif
