#include "self_attention_nvidia.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cmath>

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
__global__ void softmax_kernel(T* x, int N, int D) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        T* x_row = x + row * D;
        float max_val = -1e30f;
        for (int i = 0; i < D; ++i) {
            float val = static_cast<float>(x_row[i]);
            if (val > max_val) max_val = val;
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < D; ++i) {
            float val = static_cast<float>(x_row[i]);
            float res = expf(val - max_val);
            x_row[i] = static_cast<T>(res);
            sum_exp += res;
        }
        float inv_sum = 1.0f / sum_exp;
        for (int i = 0; i < D; ++i) {
            float val = static_cast<float>(x_row[i]);
            x_row[i] = static_cast<T>(val * inv_sum);
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    int nhead = q->shape()[1];
    int head_dim = q->shape()[2];
    int seq_len = k->shape()[0];
    int nkv_head = k->shape()[1];
    
    if (nhead % nkv_head != 0) throw std::runtime_error("SelfAttention: GQA mismatch");
    int group_size = nhead / nkv_head;
    
    CublasHandle handle;
    float alpha = scale, beta = 0.0f, one = 1.0f, zero = 0.0f;
    
    float* scores_dev;
    cudaMalloc(&scores_dev, nhead * seq_len * sizeof(float));
    
    const float* q_ptr = (const float*)q->data();
    const float* k_ptr = (const float*)k->data();
    const float* v_ptr = (const float*)v->data();
    float* out_ptr = (float*)attn_val->data();
    
    // Q * K^T
    for (int h = 0; h < nhead; ++h) {
        int kv_h = h / group_size;
        cublasSgemv(handle.get(), CUBLAS_OP_T, head_dim, seq_len, &alpha, 
            k_ptr + kv_h * head_dim, nkv_head * head_dim, 
            q_ptr + h * head_dim, 1, &beta, scores_dev + h * seq_len, 1);
    }
    
    // Softmax
    int threads = 256;
    int blocks = (nhead + threads - 1) / threads;
    softmax_kernel<float><<<blocks, threads>>>(scores_dev, nhead, seq_len);
    
    // Scores * V
    for (int h = 0; h < nhead; ++h) {
        int kv_h = h / group_size;
        cublasSgemv(handle.get(), CUBLAS_OP_N, head_dim, seq_len, &one, 
            v_ptr + kv_h * head_dim, nkv_head * head_dim, 
            scores_dev + h * seq_len, 1, &zero, out_ptr + h * head_dim, 1);
    }
    
    cudaFree(scores_dev);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("Kernel launch failed");
}
}
