// CUDA scatter kernel for the paged KV pool. Writes packed K_new / V_new
// rows into discrete physical pages indicated by slot_mapping.

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace llaisys::paged {

template <typename T>
__global__ void scatter_kv_kernel(
    T *big_k, T *big_v,
    const T *src_k, const T *src_v,
    const int32_t *slot_mapping,
    int n_tokens, int row_elems, int page_size) {
    // Grid: blockIdx.y = token index, blockIdx.x = chunk of row_elems
    // Threads in a block cooperate to copy one row.
    const int tok = blockIdx.y;
    if (tok >= n_tokens) return;

    const int32_t slot = slot_mapping[tok];
    const int page_id = (slot >> 16) & 0xFFFF;
    const int offset = slot & 0xFFFF;

    const int dst_row = page_id * page_size + offset;
    const T *sk = src_k + tok * row_elems;
    const T *sv = src_v + tok * row_elems;
    T *dk = big_k + dst_row * row_elems;
    T *dv = big_v + dst_row * row_elems;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < row_elems; i += stride) {
        dk[i] = sk[i];
        dv[i] = sv[i];
    }
}

void scatter_kv_cuda(
    void *big_k, void *big_v,
    const void *src_k, const void *src_v,
    const int32_t *slot_mapping,
    size_t n_tokens, size_t nkvh, size_t dh, size_t page_size,
    size_t elem_size,
    llaisysDeviceType_t /*device*/) {
    const int row_elems = static_cast<int>(nkvh * dh);
    const dim3 threads(128);
    const dim3 blocks(
        (row_elems + threads.x - 1) / threads.x,
        static_cast<unsigned int>(n_tokens));

    if (elem_size == 4) {
        scatter_kv_kernel<float><<<blocks, threads>>>(
            static_cast<float *>(big_k), static_cast<float *>(big_v),
            static_cast<const float *>(src_k), static_cast<const float *>(src_v),
            slot_mapping,
            static_cast<int>(n_tokens), row_elems, static_cast<int>(page_size));
    } else if (elem_size == 2) {
        // F16 / BF16: 16-bit copy. Use uint16_t since byte-wise semantics are
        // identical and we don't need to interpret values.
        scatter_kv_kernel<uint16_t><<<blocks, threads>>>(
            static_cast<uint16_t *>(big_k), static_cast<uint16_t *>(big_v),
            static_cast<const uint16_t *>(src_k), static_cast<const uint16_t *>(src_v),
            slot_mapping,
            static_cast<int>(n_tokens), row_elems, static_cast<int>(page_size));
    } else {
        // Fallback: byte-wise copy via 1-byte template instantiation.
        scatter_kv_kernel<uint8_t><<<blocks, threads>>>(
            static_cast<uint8_t *>(big_k), static_cast<uint8_t *>(big_v),
            static_cast<const uint8_t *>(src_k), static_cast<const uint8_t *>(src_v),
            slot_mapping,
            static_cast<int>(n_tokens), row_elems * static_cast<int>(elem_size),
            static_cast<int>(page_size));
    }
}

} // namespace llaisys::paged
