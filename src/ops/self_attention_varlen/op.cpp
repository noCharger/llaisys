#include "op.hpp"

#include "../self_attention/op.hpp"
#include "../common.hpp"

#include <vector>

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_varlen_nvidia.hpp"
#include "../../core/llaisys_core.hpp"
#endif

namespace llaisys::ops {
namespace {

inline void validate_varlen(const tensor_t &attn_val,
                            const tensor_t &q,
                            const std::vector<tensor_t> &k_blocks,
                            const std::vector<tensor_t> &v_blocks,
                            const int32_t *cu_seqlens_q,
                            int32_t batch) {
    ASSERT(batch > 0, "SelfAttentionVarlen: batch must be positive.");
    ASSERT(static_cast<int32_t>(k_blocks.size()) == batch,
           "SelfAttentionVarlen: k_blocks size must equal batch.");
    ASSERT(static_cast<int32_t>(v_blocks.size()) == batch,
           "SelfAttentionVarlen: v_blocks size must equal batch.");
    ASSERT(cu_seqlens_q != nullptr,
           "SelfAttentionVarlen: cu_seqlens_q must not be null.");
    ASSERT(cu_seqlens_q[0] == 0,
           "SelfAttentionVarlen: cu_seqlens_q[0] must be 0.");

    ASSERT(q->ndim() == 3, "SelfAttentionVarlen: q must be 3D [total_q, nh, dh].");
    ASSERT(attn_val->ndim() == 3,
           "SelfAttentionVarlen: attn_val must be 3D [total_q, nh, dv].");

    const size_t total_q = static_cast<size_t>(cu_seqlens_q[batch]);
    ASSERT(q->shape()[0] == total_q,
           "SelfAttentionVarlen: q seqlen must equal cu_seqlens_q[batch].");
    ASSERT(attn_val->shape()[0] == total_q,
           "SelfAttentionVarlen: attn_val seqlen must equal cu_seqlens_q[batch].");
    ASSERT(attn_val->shape()[1] == q->shape()[1],
           "SelfAttentionVarlen: nh mismatch between q and attn_val.");

    for (int32_t r = 0; r < batch; ++r) {
        ASSERT(cu_seqlens_q[r + 1] >= cu_seqlens_q[r],
               "SelfAttentionVarlen: cu_seqlens_q must be non-decreasing.");
        const auto &kb = k_blocks[r];
        const auto &vb = v_blocks[r];
        ASSERT(kb && vb, "SelfAttentionVarlen: per-request KV block must not be null.");
        ASSERT(kb->ndim() == 3 && vb->ndim() == 3,
               "SelfAttentionVarlen: KV blocks must be 3D.");

        const int32_t q_len = cu_seqlens_q[r + 1] - cu_seqlens_q[r];
        const size_t k_len = kb->shape()[0];
        ASSERT(q_len <= static_cast<int32_t>(k_len),
               "SelfAttentionVarlen: per-request q_len must be <= k_len (causal).");
        ASSERT(vb->shape()[0] == k_len,
               "SelfAttentionVarlen: V block length must match K block length.");
        ASSERT(kb->shape()[1] == vb->shape()[1],
               "SelfAttentionVarlen: nkvh mismatch between K and V blocks.");
    }
}

} // anonymous namespace

void self_attention_varlen(tensor_t attn_val,
                           tensor_t q,
                           const std::vector<tensor_t> &k_blocks,
                           const std::vector<tensor_t> &v_blocks,
                           const int32_t *cu_seqlens_q,
                           int32_t batch,
                           float scale) {
    validate_varlen(attn_val, q, k_blocks, v_blocks, cu_seqlens_q, batch);

#ifdef ENABLE_NVIDIA_API
    if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
        nvidia::self_attention_varlen(attn_val, q, k_blocks, v_blocks,
                                      cu_seqlens_q, batch, scale);
        return;
    }
#endif

    // CPU fallback: dispatch each request to single-sequence self_attention.
    for (int32_t r = 0; r < batch; ++r) {
        const size_t start = static_cast<size_t>(cu_seqlens_q[r]);
        const size_t end = static_cast<size_t>(cu_seqlens_q[r + 1]);
        if (end == start) {
            continue; // empty request slot
        }

        tensor_t q_slice = q->slice(0, start, end);
        tensor_t out_slice = attn_val->slice(0, start, end);
        self_attention(out_slice, q_slice, k_blocks[r], v_blocks[r], scale);
    }
}

} // namespace llaisys::ops
