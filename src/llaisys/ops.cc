#include "llaisys/ops.h"

#include "llaisys_tensor.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/self_attention_varlen/op.hpp"
#include "../ops/self_attention_paged/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/random_sample/op.hpp"

__C {
    void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b) {
        llaisys::ops::add(c->tensor, a->tensor, b->tensor);
    }
    void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals) {
        llaisys::ops::argmax(max_idx->tensor, max_val->tensor, vals->tensor);
    }
    void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight) {
        llaisys::ops::embedding(out->tensor, index->tensor, weight->tensor);
    }
    void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias) {
        llaisys::ops::linear(out->tensor, in->tensor, weight->tensor, bias->tensor);
    }
    void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in) {
        llaisys::ops::rearrange(out->tensor, in->tensor);
    }
    void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps) {
        llaisys::ops::rms_norm(out->tensor, in->tensor, weight->tensor, eps);
    }
    void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta) {
        llaisys::ops::rope(out->tensor, in->tensor, pos_ids->tensor, theta);
    }
    void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale) {
        llaisys::ops::self_attention(attn_val->tensor, q->tensor, k->tensor, v->tensor, scale);
    }
    void llaisysSelfAttentionVarlen(
        llaisysTensor_t attn_val,
        llaisysTensor_t q,
        llaisysTensor_t *k_blocks,
        llaisysTensor_t *v_blocks,
        const int32_t *cu_seqlens_q,
        int32_t batch,
        float scale) {
        try {
            std::vector<llaisys::tensor_t> ks, vs;
            ks.reserve(static_cast<size_t>(batch));
            vs.reserve(static_cast<size_t>(batch));
            for (int32_t r = 0; r < batch; ++r) {
                ks.push_back(k_blocks[r]->tensor);
                vs.push_back(v_blocks[r]->tensor);
            }
            llaisys::ops::self_attention_varlen(
                attn_val->tensor, q->tensor, ks, vs, cu_seqlens_q, batch, scale);
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] llaisysSelfAttentionVarlen: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[ERROR] llaisysSelfAttentionVarlen: unknown exception" << std::endl;
        }
    }
    void llaisysSelfAttentionPaged(
        llaisysTensor_t attn_val,
        llaisysTensor_t q,
        llaisysTensor_t big_k,
        llaisysTensor_t big_v,
        const int32_t *block_tables,
        const int32_t *block_table_lens,
        const int32_t *cu_seqlens_q,
        const int32_t *kv_lens,
        int32_t batch,
        int32_t page_size,
        float scale) {
        try {
            llaisys::ops::self_attention_paged(
                attn_val->tensor, q->tensor,
                big_k->tensor, big_v->tensor,
                block_tables, block_table_lens, cu_seqlens_q, kv_lens,
                batch, page_size, scale);
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] llaisysSelfAttentionPaged: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[ERROR] llaisysSelfAttentionPaged: unknown exception" << std::endl;
        }
    }
    void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up) {
        llaisys::ops::swiglu(out->tensor, gate->tensor, up->tensor);
    }
    void llaisysRandomSample(llaisysTensor_t out_token, llaisysTensor_t logits, float temp, float top_p, int top_k) {
        llaisys::ops::random_sample(out_token->tensor, logits->tensor, temp, top_p, top_k);
    }
    void llaisysRandomSampleBatch(
        llaisysTensor_t out_tokens,
        llaisysTensor_t logits,
        const float *temps,
        const float *top_ps,
        const int *top_ks) {
        try {
            llaisys::ops::random_sample_batch(
                out_tokens->tensor, logits->tensor, temps, top_ps, top_ks);
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] llaisysRandomSampleBatch: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[ERROR] llaisysRandomSampleBatch: unknown exception" << std::endl;
        }
    }
    void llaisysArgmaxBatch(llaisysTensor_t out_indices, llaisysTensor_t logits) {
        try {
            llaisys::ops::argmax_batch(out_indices->tensor, logits->tensor);
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] llaisysArgmaxBatch: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[ERROR] llaisysArgmaxBatch: unknown exception" << std::endl;
        }
    }
}
