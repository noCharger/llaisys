#include "rms_norm_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void rms_norm_kernel(T* out, const T* in, const T* weight, float eps, int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const T* in_row = in + row * D;
    T* out_row = out + row * D;

    float sum_sq = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = static_cast<float>(in_row[i]);
        sum_sq += val * val;
    }

    extern __shared__ float sdata[];
    sdata[tid] = sum_sq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / D + eps);

    for (int i = tid; i < D; i += blockDim.x) {
        out_row[i] = static_cast<T>(static_cast<float>(in_row[i]) * inv_rms * static_cast<float>(weight[i]));
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    int N = in->shape()[0];
    int D = in->shape()[1];
    int threads = 256;
    size_t smem = threads * sizeof(float);
    
    auto dtype = out->dtype();
    if (dtype == LLAISYS_DTYPE_F32) {
        rms_norm_kernel<float><<<N, threads, smem>>>((float*)out->data(), (const float*)in->data(), (const float*)weight->data(), eps, D);
    } else if (dtype == LLAISYS_DTYPE_F16) {
        rms_norm_kernel<half><<<N, threads, smem>>>((half*)out->data(), (const half*)in->data(), (const half*)weight->data(), eps, D);
    } else if (dtype == LLAISYS_DTYPE_BF16) {
        rms_norm_kernel<__nv_bfloat16><<<N, threads, smem>>>((__nv_bfloat16*)out->data(), (const __nv_bfloat16*)in->data(), (const __nv_bfloat16*)weight->data(), eps, D);
    } else {
        throw std::runtime_error("RMSNorm NVIDIA: Unsupported data type");
    }
    
    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}
}
