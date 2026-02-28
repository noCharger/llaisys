#include "rope_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void rope_kernel(T* out, const T* in, const int64_t* pos_ids, float theta, int nhead, int head_dim, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_pairs = seq_len * nhead * half_dim;
    
    if (idx < total_pairs) {
        int d = idx % half_dim;
        int rem = idx / half_dim;
        int h = rem % nhead;
        int s = rem / nhead;
        
        int head_offset = (s * nhead + h) * head_dim;
        int idx1 = head_offset + d;
        int idx2 = head_offset + d + half_dim;
        
        float x1 = static_cast<float>(in[idx1]);
        float x2 = static_cast<float>(in[idx2]);
        
        int pos = pos_ids[s];
        float freq = 1.0f / powf(theta, (float)(2 * d) / head_dim);
        float alpha = (float)pos * freq;
        
        float cos_val = cosf(alpha);
        float sin_val = sinf(alpha);
        
        out[idx1] = static_cast<T>(x1 * cos_val - x2 * sin_val);
        out[idx2] = static_cast<T>(x1 * sin_val + x2 * cos_val);
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    int seq_len = in->shape()[0];
    int nhead = in->shape()[1];
    int head_dim = in->shape()[2];
    
    int total_pairs = seq_len * nhead * head_dim / 2;
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    
    if (out->dtype() == LLAISYS_DTYPE_F32) {
        rope_kernel<float><<<blocks, threads>>>((float*)out->data(), (const float*)in->data(), (const int64_t*)pos_ids->data(), theta, nhead, head_dim, seq_len);
    } else if (out->dtype() == LLAISYS_DTYPE_F16) {
        rope_kernel<half><<<blocks, threads>>>((half*)out->data(), (const half*)in->data(), (const int64_t*)pos_ids->data(), theta, nhead, head_dim, seq_len);
    } else if (out->dtype() == LLAISYS_DTYPE_BF16) {
        rope_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out->data(), (const __nv_bfloat16*)in->data(), (const int64_t*)pos_ids->data(), theta, nhead, head_dim, seq_len);
    } else {
        throw std::runtime_error("RoPE NVIDIA: Unsupported data type");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("Kernel launch failed");
}
}
