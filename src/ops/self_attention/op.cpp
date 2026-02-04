#include "op.hpp"
#include "../common.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

namespace llaisys::ops {
namespace {

inline void validate_self_attention_tensors(const tensor_t& attn_val,
                                            const tensor_t& q,
                                            const tensor_t& k,
                                            const tensor_t& v) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);

    ASSERT(attn_val->ndim() == 3, "SelfAttention: attn_val must be 3D [seqlen, nhead, dv].");
    ASSERT(q->ndim() == 3, "SelfAttention: q must be 3D [seqlen, nhead, d].");
    ASSERT(k->ndim() == 3, "SelfAttention: k must be 3D [total_len, nkvhead, d].");
    ASSERT(v->ndim() == 3, "SelfAttention: v must be 3D [total_len, nkvhead, dv].");

    const size_t seqlen    = q->shape()[0];
    const size_t nhead     = q->shape()[1];
    const size_t d         = q->shape()[2];
    
    const size_t total_len = k->shape()[0];
    const size_t nkvhead   = k->shape()[1];
    // k->shape()[2] checked against d below
    const size_t dv        = v->shape()[2];

    // Q and K head dimension match
    ASSERT(k->shape()[2] == d, "SelfAttention: K head dim must match Q head dim.");
    
    // Q and Output seqlen/nhead match
    ASSERT(attn_val->shape()[0] == seqlen, "SelfAttention: Output seqlen mismatch.");
    ASSERT(attn_val->shape()[1] == nhead,  "SelfAttention: Output nhead mismatch.");
    ASSERT(attn_val->shape()[2] == dv,     "SelfAttention: Output head dim must match V head dim.");

    // K and V length match
    ASSERT(v->shape()[0] == total_len, "SelfAttention: V total_len must match K total_len.");
    ASSERT(v->shape()[1] == nkvhead,   "SelfAttention: V nkvhead must match K nkvhead.");

    // Grouped Query Attention (GQA) check
    ASSERT(nhead % nkvhead == 0, "SelfAttention: nhead must be divisible by nkvhead (GQA).");

    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    ASSERT(llaisys::ops::common::allContiguous(q, k, v), 
           "SelfAttention: all tensors must be contiguous.");
    ASSERT(attn_val->isContiguous(), "SelfAttention: attn_val must be contiguous."); 
}

template <typename T>
void self_attention_cpu(const tensor_t& attn_val,
                        const tensor_t& q,
                        const tensor_t& k,
                        const tensor_t& v,
                        float scale) {
    const size_t seqlen    = q->shape()[0];
    const size_t nhead     = q->shape()[1];
    const size_t d         = q->shape()[2];
    
    const size_t total_len = k->shape()[0];
    const size_t nkvhead   = k->shape()[1];
    const size_t dv        = v->shape()[2];

    // GQA factor
    const size_t group_size = nhead / nkvhead;

    // Pointers
    const T* q_ptr = reinterpret_cast<const T*>(q->data());
    const T* k_ptr = reinterpret_cast<const T*>(k->data());
    const T* v_ptr = reinterpret_cast<const T*>(v->data());
    T* out_ptr     = reinterpret_cast<T*>(attn_val->data());

    // Strides for iterating through time t in K and V
    // K/V shape is [total_len, nkvhead, dim]. 
    // Jumping t+1 means skipping (nkvhead * dim) elements.
    const size_t stride_k_t = nkvhead * d;
    const size_t stride_v_t = nkvhead * dv;

    // Calculate past length to determine global position
    // If input is a chunk (seqlen), it appends to the end of total_len.
    const size_t past_len = total_len - seqlen;

    // Buffer for attention scores (softmax logits).
    // Max size needed per head is `total_len`.
    std::vector<float> logits(total_len);

    // Buffer for accumulating weighted sum to avoid repeated allocation in inner loop
    std::vector<float> acc(dv);
    
    for (size_t i = 0; i < seqlen; ++i) {
        // Global position of the current query token
        const size_t current_pos = past_len + i;

        for (size_t h = 0; h < nhead; ++h) {
            // Identify which KV head to use (GQA)
            const size_t kv_h = h / group_size;

            // Compute base offsets for K and V for this specific head (kv_h)
            // The head offset within a time-step block is (kv_h * dim)
            const size_t offset_k_h = kv_h * d;
            const size_t offset_v_h = kv_h * dv;

            // Pointer to current query vector: q[i, h, :]
            const T* q_vec = q_ptr + (i * nhead + h) * d;

            // 1. Compute Attention Scores: A = Q * K^T * scale
            // Optimization: Only iterate up to current_pos (implicit causal mask).
            float max_logit = -std::numeric_limits<float>::infinity();

            const T* k_base_ptr = k_ptr + offset_k_h; // Points to K[0, kv_h, :]

            for (size_t t = 0; t <= current_pos; ++t) {
                const T* k_vec = k_base_ptr + t * stride_k_t;

                float dot = 0.0f;
                for (size_t j = 0; j < d; ++j) {
                    float q_val = llaisys::ops::common::to_float(q_vec[j]);
                    float k_val = llaisys::ops::common::to_float(k_vec[j]);
                    dot += q_val * k_val;
                }
                
                float score = dot * scale;
                logits[t] = score;
                if (score > max_logit) {
                    max_logit = score;
                }
            }

            // 2. Softmax: exp(x - max) / sum(exp)
            float sum_exp = 0.0f;
            for (size_t t = 0; t <= current_pos; ++t) {
                // std::exp is overloaded for float in <cmath>
                float val = std::exp(logits[t] - max_logit);
                logits[t] = val; 
                sum_exp += val;
            }
            
            // Avoid division by zero (though unlikely with softmax)
            const float inv_sum_exp = 1.0f / sum_exp;

            // 3. Weighted Sum: Y = Softmax(A) * V
            // Initialize accumulator
            std::fill(acc.begin(), acc.end(), 0.0f);

            const T* v_base_ptr = v_ptr + offset_v_h; // Points to V[0, kv_h, :]

            for (size_t t = 0; t <= current_pos; ++t) {
                const float weight = logits[t] * inv_sum_exp;
                const T* v_vec = v_base_ptr + t * stride_v_t;

                for (size_t j = 0; j < dv; ++j) {
                    float v_val = llaisys::ops::common::to_float(v_vec[j]);
                    acc[j] += weight * v_val;
                }
            }

            // Store result: out[i, h, :]
            T* out_vec = out_ptr + (i * nhead + h) * dv;
            for (size_t j = 0; j < dv; ++j) {
                out_vec[j] = llaisys::ops::common::from_float<T>(acc[j]);
            }
        }
    }
}

} // anonymous namespace

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    validate_self_attention_tensors(attn_val, q, k, v);

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (attn_val->dtype()) {
            case LLAISYS_DTYPE_F32:
                self_attention_cpu<float>(attn_val, q, k, v, scale);
                return;
            case LLAISYS_DTYPE_F16:
                self_attention_cpu<llaisys::fp16_t>(attn_val, q, k, v, scale);
                return;
            case LLAISYS_DTYPE_BF16:
                self_attention_cpu<llaisys::bf16_t>(attn_val, q, k, v, scale);
                return;
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
        }
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops