#include "self_attention_varlen_nvidia.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

#include <cmath>
#include <stdexcept>

#include "utils/cuda_check.cuh"

namespace llaisys::ops::nvidia {
using namespace llaisys::utils;

// Local copy of the softmax kernel from self_attention/nvidia (sharing
// __global__ functions across .cu units needs RDC).
template <typename T>
__global__ void attention_softmax_kernel_varlen(T *scores, int q_len, int kv_len,
                                                int nhead, float scale) {
    const int h = blockIdx.y;
    const int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= q_len || h >= nhead) return;

    T *row = scores + (h * q_len + q_idx) * kv_len;
    const int mask_limit = q_idx + (kv_len - q_len);

    float max_val = -FLT_MAX;
    for (int i = 0; i <= mask_limit; ++i) {
        float val = static_cast<float>(row[i]) * scale;
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < kv_len; ++i) {
        if (i <= mask_limit) {
            float val = expf((static_cast<float>(row[i]) * scale) - max_val);
            row[i] = static_cast<T>(val);
            sum_exp += val;
        } else {
            row[i] = static_cast<T>(0.0f);
        }
    }

    const float inv_sum = 1.0f / (sum_exp + 1e-9f);
    for (int i = 0; i < kv_len; ++i) {
        row[i] = static_cast<T>(static_cast<float>(row[i]) * inv_sum);
    }
}

template <typename T>
static size_t max_scratch_bytes(int nhead, const std::vector<tensor_t> &k_blocks,
                                const int32_t *cu_seqlens_q, int32_t batch) {
    size_t best = 0;
    for (int32_t r = 0; r < batch; ++r) {
        const int q_len = cu_seqlens_q[r + 1] - cu_seqlens_q[r];
        const int kv_len = static_cast<int>(k_blocks[r]->shape()[0]);
        const size_t bytes = static_cast<size_t>(nhead) * q_len * kv_len * sizeof(T);
        if (bytes > best) best = bytes;
    }
    return best;
}

template <typename T, cudaDataType_t CudaType>
static void self_attention_varlen_impl(tensor_t attn_val, tensor_t q,
                                       const std::vector<tensor_t> &k_blocks,
                                       const std::vector<tensor_t> &v_blocks,
                                       const int32_t *cu_seqlens_q,
                                       int32_t batch, float scale) {
    const int nhead = static_cast<int>(q->shape()[1]);
    const int head_dim = static_cast<int>(q->shape()[2]);
    const int nkvh = static_cast<int>(k_blocks[0]->shape()[1]);
    const int heads_per_group = nhead / nkvh;

    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f, beta = 0.0f;

    const size_t scratch_bytes =
        max_scratch_bytes<T>(nhead, k_blocks, cu_seqlens_q, batch);
    T *scores_dev = static_cast<T *>(get_scratch_buffer(scratch_bytes));

    T *q_base = static_cast<T *>(q->data());
    T *out_base = static_cast<T *>(attn_val->data());

    for (int32_t r = 0; r < batch; ++r) {
        const int q_off = cu_seqlens_q[r];
        const int q_len = cu_seqlens_q[r + 1] - q_off;
        if (q_len == 0) continue;

        const int kv_len = static_cast<int>(k_blocks[r]->shape()[0]);
        T *q_ptr = q_base + static_cast<size_t>(q_off) * nhead * head_dim;
        T *out_ptr = out_base + static_cast<size_t>(q_off) * nhead * head_dim;
        T *k_ptr = static_cast<T *>(k_blocks[r]->data());
        T *v_ptr = static_cast<T *>(v_blocks[r]->data());

        // Q * K^T -> scores
        for (int g = 0; g < nkvh; ++g) {
            cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                kv_len, q_len, head_dim,
                &alpha,
                static_cast<const void *>(k_ptr + g * head_dim),
                CudaType, nkvh * head_dim, 0,
                static_cast<const void *>(q_ptr + g * heads_per_group * head_dim),
                CudaType, nhead * head_dim, head_dim,
                &beta,
                static_cast<void *>(scores_dev + g * heads_per_group * q_len * kv_len),
                CudaType, kv_len, q_len * kv_len,
                heads_per_group,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }

        // softmax + causal mask, in place
        dim3 threads(32);
        dim3 blocks((q_len + 31) / 32, nhead);
        attention_softmax_kernel_varlen<T><<<blocks, threads>>>(
            scores_dev, q_len, kv_len, nhead, scale);

        // scores * V -> out
        for (int g = 0; g < nkvh; ++g) {
            cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                head_dim, q_len, kv_len,
                &alpha,
                static_cast<const void *>(v_ptr + g * head_dim),
                CudaType, nkvh * head_dim, 0,
                static_cast<const void *>(scores_dev + g * heads_per_group * q_len * kv_len),
                CudaType, kv_len, q_len * kv_len,
                &beta,
                static_cast<void *>(out_ptr + g * heads_per_group * head_dim),
                CudaType, nhead * head_dim, head_dim,
                heads_per_group,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
    }
}

void self_attention_varlen(tensor_t attn_val, tensor_t q,
                           const std::vector<tensor_t> &k_blocks,
                           const std::vector<tensor_t> &v_blocks,
                           const int32_t *cu_seqlens_q, int32_t batch,
                           float scale) {
    switch (q->dtype()) {
        case LLAISYS_DTYPE_F32:
            self_attention_varlen_impl<float, CUDA_R_32F>(
                attn_val, q, k_blocks, v_blocks, cu_seqlens_q, batch, scale);
            return;
        case LLAISYS_DTYPE_F16:
            self_attention_varlen_impl<half, CUDA_R_16F>(
                attn_val, q, k_blocks, v_blocks, cu_seqlens_q, batch, scale);
            return;
        case LLAISYS_DTYPE_BF16:
            self_attention_varlen_impl<__nv_bfloat16, CUDA_R_16BF>(
                attn_val, q, k_blocks, v_blocks, cu_seqlens_q, batch, scale);
            return;
        default:
            throw std::runtime_error("Unsupported dtype for self_attention_varlen");
    }
}

} // namespace llaisys::ops::nvidia
