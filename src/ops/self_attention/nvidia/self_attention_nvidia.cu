#include "self_attention_nvidia.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <cmath>
#include <float.h>

namespace llaisys::ops::nvidia {

class CublasHandle {
public:
    CublasHandle() { cublasCreate(&handle_); }
    ~CublasHandle() { cublasDestroy(handle_); }
    cublasHandle_t get() { return handle_; }
private:
    cublasHandle_t handle_;
};

template <typename T>
__global__ void attention_softmax_kernel(T* scores, int q_len, int kv_len, int nhead, float scale) {
    int h = blockIdx.y; 
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (q_idx >= q_len || h >= nhead) return;

    T* row = scores + (h * q_len + q_idx) * kv_len;
    
    // Causal masking
    int mask_limit = q_idx + (kv_len - q_len);

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

    float inv_sum = 1.0f / (sum_exp + 1e-9f);
    for (int i = 0; i < kv_len; ++i) {
        row[i] = static_cast<T>(static_cast<float>(row[i]) * inv_sum);
    }
}

template <typename T, cudaDataType_t CudaType>
void self_attention_impl(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    int q_len = q->shape()[0];
    int nhead = q->shape()[1];
    int head_dim = q->shape()[2];
    int kv_len = k->shape()[0];
    int nkvh = k->shape()[1];
    
    CublasHandle handle;
    float alpha = 1.0f, beta = 0.0f;

    T* scores_dev;
    cudaMalloc(&scores_dev, nhead * q_len * kv_len * sizeof(T));

    // Q * K^T
    for (int g = 0; g < nkvh; ++g) {
        int heads_per_group = nhead / nkvh;
        cublasGemmStridedBatchedEx(handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_N, 
            kv_len, q_len, head_dim,
            &alpha,
            (const void*)((T*)k->data() + (g * head_dim)), CudaType, nkvh * head_dim, 0, 
            (const void*)((T*)q->data() + (g * heads_per_group * head_dim)), CudaType, nhead * head_dim, head_dim,
            &beta,
            (void*)(scores_dev + (g * heads_per_group * q_len * kv_len)), CudaType, kv_len, q_len * kv_len,
            heads_per_group,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
    }

    dim3 threads(32);
    dim3 blocks((q_len + 31) / 32, nhead);
    attention_softmax_kernel<T><<<blocks, threads>>>(scores_dev, q_len, kv_len, nhead, scale);

    // Scores * V
    for (int g = 0; g < nkvh; ++g) {
        int heads_per_group = nhead / nkvh;
        cublasGemmStridedBatchedEx(handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim, q_len, kv_len,
            &alpha,
            (const void*)((T*)v->data() + (g * head_dim)), CudaType, nkvh * head_dim, 0,
            (const void*)(scores_dev + (g * heads_per_group * q_len * kv_len)), CudaType, kv_len, q_len * kv_len,
            &beta,
            (void*)((T*)attn_val->data() + (g * heads_per_group * head_dim)), CudaType, nhead * head_dim, head_dim,
            heads_per_group,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
    }

    cudaFree(scores_dev);
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    auto dtype = q->dtype();

    if (dtype == LLAISYS_DTYPE_F32) {
        self_attention_impl<float, CUDA_R_32F>(attn_val, q, k, v, scale);
    } else if (dtype == LLAISYS_DTYPE_F16) {
        self_attention_impl<half, CUDA_R_16F>(attn_val, q, k, v, scale);
    } else if (dtype == LLAISYS_DTYPE_BF16) {
        self_attention_impl<__nv_bfloat16, CUDA_R_16BF>(attn_val, q, k, v, scale);
    } else {
        throw std::runtime_error("Unsupported dtype for self_attention");
    }
}

}