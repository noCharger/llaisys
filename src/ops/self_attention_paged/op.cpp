#include "op.hpp"

#include "../self_attention/op.hpp"
#include "../common.hpp"
#include "../../utils/check.hpp"

#include <cstdint>
#include <vector>

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_paged_nvidia.hpp"
#include "../../core/llaisys_core.hpp"
#endif

namespace llaisys::ops {
namespace {

inline void validate_paged(const tensor_t &attn_val,
                           const tensor_t &q,
                           const tensor_t &big_k,
                           const tensor_t &big_v,
                           const int32_t *block_tables,
                           const int32_t *block_table_lens,
                           const int32_t *cu_seqlens_q,
                           const int32_t *kv_lens,
                           int32_t batch, int32_t page_size) {
    ASSERT(batch > 0, "SelfAttentionPaged: batch must be positive.");
    ASSERT(page_size > 0, "SelfAttentionPaged: page_size must be positive.");
    ASSERT(block_tables && block_table_lens && cu_seqlens_q && kv_lens,
           "SelfAttentionPaged: all metadata pointers must be non-null.");
    ASSERT(cu_seqlens_q[0] == 0,
           "SelfAttentionPaged: cu_seqlens_q[0] must be 0.");

    ASSERT(q->ndim() == 3, "SelfAttentionPaged: q must be 3D [total_q, nh, dh].");
    ASSERT(attn_val->ndim() == 3,
           "SelfAttentionPaged: attn_val must be 3D [total_q, nh, dv].");
    ASSERT(big_k->ndim() == 3,
           "SelfAttentionPaged: big_k must be 3D [n_pages*page_size, nkvh, dh].");
    ASSERT(big_v->ndim() == 3,
           "SelfAttentionPaged: big_v must be 3D [n_pages*page_size, nkvh, dh].");

    const size_t total_q = static_cast<size_t>(cu_seqlens_q[batch]);
    ASSERT(q->shape()[0] == total_q,
           "SelfAttentionPaged: q seqlen must equal cu_seqlens_q[batch].");
    ASSERT(attn_val->shape()[0] == total_q,
           "SelfAttentionPaged: attn_val seqlen must equal cu_seqlens_q[batch].");

    for (int32_t r = 0; r < batch; ++r) {
        ASSERT(cu_seqlens_q[r + 1] >= cu_seqlens_q[r],
               "SelfAttentionPaged: cu_seqlens_q must be non-decreasing.");
        const int32_t q_len = cu_seqlens_q[r + 1] - cu_seqlens_q[r];
        const int32_t kv_len = kv_lens[r];
        ASSERT(q_len <= kv_len,
               "SelfAttentionPaged: q_len must be <= kv_len for each request.");
        ASSERT(block_table_lens[r] * page_size >= kv_len,
               "SelfAttentionPaged: block_table not big enough for kv_len.");
    }
}

// Gather KV from pool pages into a contiguous [kv_len, nkvh, dh] buffer.
template <typename T>
void gather_kv_cpu(const T *big_buf,
                  const int32_t *block_table, int32_t npages,
                  int32_t kv_len, int32_t page_size,
                  int32_t nkvh, int32_t dh,
                  T *out_buf) {
    const size_t row_elems = static_cast<size_t>(nkvh) * dh;
    int32_t copied = 0;
    for (int32_t p = 0; p < npages && copied < kv_len; ++p) {
        const int32_t page_id = block_table[p];
        const int32_t take = std::min(page_size, kv_len - copied);
        const T *src = big_buf + (static_cast<size_t>(page_id) * page_size) * row_elems;
        T *dst = out_buf + static_cast<size_t>(copied) * row_elems;
        std::memcpy(dst, src,
                    static_cast<size_t>(take) * row_elems * sizeof(T));
        copied += take;
    }
}

template <typename T>
void self_attention_paged_cpu_impl(
    tensor_t attn_val, tensor_t q,
    tensor_t big_k, tensor_t big_v,
    const int32_t *block_tables, const int32_t *block_table_lens,
    const int32_t *cu_seqlens_q, const int32_t *kv_lens,
    int32_t batch, int32_t page_size, float scale) {
    const int32_t dh = static_cast<int32_t>(q->shape()[2]);
    const int32_t nkvh = static_cast<int32_t>(big_k->shape()[1]);

    const T *big_k_ptr = reinterpret_cast<const T *>(big_k->data());
    const T *big_v_ptr = reinterpret_cast<const T *>(big_v->data());

    int32_t bt_offset = 0;
    for (int32_t r = 0; r < batch; ++r) {
        const int32_t q_off = cu_seqlens_q[r];
        const int32_t q_end = cu_seqlens_q[r + 1];
        const int32_t q_len = q_end - q_off;
        if (q_len == 0) {
            bt_offset += block_table_lens[r];
            continue;
        }
        const int32_t kv_len = kv_lens[r];
        const int32_t npages = block_table_lens[r];

        std::vector<T> tmp_k(static_cast<size_t>(kv_len) * nkvh * dh);
        std::vector<T> tmp_v(static_cast<size_t>(kv_len) * nkvh * dh);
        gather_kv_cpu<T>(big_k_ptr, block_tables + bt_offset, npages,
                         kv_len, page_size, nkvh, dh, tmp_k.data());
        gather_kv_cpu<T>(big_v_ptr, block_tables + bt_offset, npages,
                         kv_len, page_size, nkvh, dh, tmp_v.data());

        const std::vector<size_t> shape_k = {static_cast<size_t>(kv_len),
                                              static_cast<size_t>(nkvh),
                                              static_cast<size_t>(dh)};
        tensor_t k_t = Tensor::create(shape_k, big_k->dtype(),
                                       big_k->deviceType(), big_k->deviceId());
        tensor_t v_t = Tensor::create(shape_k, big_v->dtype(),
                                       big_v->deviceType(), big_v->deviceId());
        std::memcpy(k_t->data(), tmp_k.data(),
                    tmp_k.size() * sizeof(T));
        std::memcpy(v_t->data(), tmp_v.data(),
                    tmp_v.size() * sizeof(T));

        tensor_t q_slice = q->slice(0, static_cast<size_t>(q_off),
                                       static_cast<size_t>(q_end));
        tensor_t out_slice = attn_val->slice(0, static_cast<size_t>(q_off),
                                              static_cast<size_t>(q_end));
        self_attention(out_slice, q_slice, k_t, v_t, scale);

        bt_offset += npages;
    }
}

} // anonymous namespace

void self_attention_paged(tensor_t attn_val,
                          tensor_t q,
                          tensor_t big_k,
                          tensor_t big_v,
                          const int32_t *block_tables,
                          const int32_t *block_table_lens,
                          const int32_t *cu_seqlens_q,
                          const int32_t *kv_lens,
                          int32_t batch,
                          int32_t page_size,
                          float scale) {
    validate_paged(attn_val, q, big_k, big_v,
                   block_tables, block_table_lens, cu_seqlens_q, kv_lens,
                   batch, page_size);

#ifdef ENABLE_NVIDIA_API
    if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
        nvidia::self_attention_paged(
            attn_val, q, big_k, big_v,
            block_tables, block_table_lens, cu_seqlens_q, kv_lens,
            batch, page_size, scale);
        return;
    }
#endif

    if (attn_val->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    switch (attn_val->dtype()) {
        case LLAISYS_DTYPE_F32:
            self_attention_paged_cpu_impl<float>(
                attn_val, q, big_k, big_v,
                block_tables, block_table_lens, cu_seqlens_q, kv_lens,
                batch, page_size, scale);
            return;
        case LLAISYS_DTYPE_F16:
            self_attention_paged_cpu_impl<llaisys::fp16_t>(
                attn_val, q, big_k, big_v,
                block_tables, block_table_lens, cu_seqlens_q, kv_lens,
                batch, page_size, scale);
            return;
        case LLAISYS_DTYPE_BF16:
            self_attention_paged_cpu_impl<llaisys::bf16_t>(
                attn_val, q, big_k, big_v,
                block_tables, block_table_lens, cu_seqlens_q, kv_lens,
                batch, page_size, scale);
            return;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
    }
}

} // namespace llaisys::ops
