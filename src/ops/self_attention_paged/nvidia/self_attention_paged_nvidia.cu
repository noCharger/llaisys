#include "self_attention_paged_nvidia.hpp"
#include "../../self_attention/nvidia/self_attention_nvidia.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "utils/cuda_check.cuh"

namespace llaisys::ops::nvidia {

// One CUDA block per (token, copy chunk) pair fills one KV row.
template <typename T>
__global__ void paged_gather_kernel(
    const T *big_buf,
    const int32_t *block_table, int32_t npages,
    int32_t kv_len, int32_t page_size,
    int32_t row_elems,
    T *out_buf) {
    const int tok = blockIdx.y;
    if (tok >= kv_len) return;

    const int page_idx = tok / page_size;
    const int offset = tok % page_size;
    if (page_idx >= npages) return;
    const int page_id = block_table[page_idx];

    const T *src = big_buf + (page_id * page_size + offset) * row_elems;
    T *dst = out_buf + tok * row_elems;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < row_elems; i += stride) {
        dst[i] = src[i];
    }
}

template <typename T>
void launch_paged_gather(
    const T *big_buf, const int32_t *block_table_dev,
    int32_t npages, int32_t kv_len, int32_t page_size,
    int32_t row_elems, T *out_buf) {
    const dim3 threads(128);
    const dim3 blocks(
        (row_elems + threads.x - 1) / threads.x,
        static_cast<unsigned int>(kv_len));
    paged_gather_kernel<T><<<blocks, threads>>>(
        big_buf, block_table_dev, npages, kv_len, page_size, row_elems, out_buf);
}

template <typename T, cudaDataType_t CudaType>
void self_attention_paged_impl(
    tensor_t attn_val, tensor_t q, tensor_t big_k, tensor_t big_v,
    const int32_t *block_tables, const int32_t *block_table_lens,
    const int32_t *cu_seqlens_q, const int32_t *kv_lens,
    int32_t batch, int32_t page_size, float scale) {
    const int nkvh = static_cast<int>(big_k->shape()[1]);
    const int dh = static_cast<int>(big_k->shape()[2]);
    const int row_elems = nkvh * dh;
    const llaisysDeviceType_t dev = attn_val->deviceType();
    const int dev_id = attn_val->deviceId();

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

        std::vector<size_t> shape_kv = {static_cast<size_t>(kv_len),
                                          static_cast<size_t>(nkvh),
                                          static_cast<size_t>(dh)};
        tensor_t k_t = Tensor::create(shape_kv, big_k->dtype(), dev, dev_id);
        tensor_t v_t = Tensor::create(shape_kv, big_v->dtype(), dev, dev_id);

        int32_t *bt_dev = nullptr;
        cudaMalloc(&bt_dev, npages * sizeof(int32_t));
        cudaMemcpyAsync(bt_dev, block_tables + bt_offset,
                        npages * sizeof(int32_t), cudaMemcpyHostToDevice);

        launch_paged_gather<T>(
            reinterpret_cast<const T *>(big_k->data()), bt_dev,
            npages, kv_len, page_size, row_elems,
            reinterpret_cast<T *>(k_t->data()));
        launch_paged_gather<T>(
            reinterpret_cast<const T *>(big_v->data()), bt_dev,
            npages, kv_len, page_size, row_elems,
            reinterpret_cast<T *>(v_t->data()));
        cudaFree(bt_dev);

        tensor_t q_slice = q->slice(0, static_cast<size_t>(q_off),
                                       static_cast<size_t>(q_end));
        tensor_t out_slice = attn_val->slice(0, static_cast<size_t>(q_off),
                                              static_cast<size_t>(q_end));

        self_attention(out_slice, q_slice, k_t, v_t, scale);

        bt_offset += npages;
    }
}

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
    switch (q->dtype()) {
        case LLAISYS_DTYPE_F32:
            self_attention_paged_impl<float, CUDA_R_32F>(
                attn_val, q, big_k, big_v,
                block_tables, block_table_lens, cu_seqlens_q, kv_lens,
                batch, page_size, scale);
            return;
        case LLAISYS_DTYPE_F16:
            self_attention_paged_impl<half, CUDA_R_16F>(
                attn_val, q, big_k, big_v,
                block_tables, block_table_lens, cu_seqlens_q, kv_lens,
                batch, page_size, scale);
            return;
        case LLAISYS_DTYPE_BF16:
            self_attention_paged_impl<__nv_bfloat16, CUDA_R_16BF>(
                attn_val, q, big_k, big_v,
                block_tables, block_table_lens, cu_seqlens_q, kv_lens,
                batch, page_size, scale);
            return;
        default:
            throw std::runtime_error("Unsupported dtype for self_attention_paged");
    }
}

} // namespace llaisys::ops::nvidia
