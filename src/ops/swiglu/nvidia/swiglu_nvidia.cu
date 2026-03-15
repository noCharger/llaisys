#include "swiglu_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <cmath>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void swiglu_kernel(T* out, const T* gate, const T* up, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float g = static_cast<float>(gate[idx]);
    float u = static_cast<float>(up[idx]);
    float sigmoid_g = 1.0f / (1.0f + expf(-g));
    out[idx] = static_cast<T>(u * (g * sigmoid_g));
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    int size = out->numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    auto dtype = out->dtype();
    if (dtype == LLAISYS_DTYPE_F32) {
        swiglu_kernel<float><<<blocks, threads>>>((float*)out->data(), (const float*)gate->data(), (const float*)up->data(), size);
    } else if (dtype == LLAISYS_DTYPE_F16) {
        swiglu_kernel<half><<<blocks, threads>>>((half*)out->data(), (const half*)gate->data(), (const half*)up->data(), size);
    } else if (dtype == LLAISYS_DTYPE_BF16) {
        swiglu_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out->data(), (const __nv_bfloat16*)gate->data(), (const __nv_bfloat16*)up->data(), size);
    } else {
        throw std::runtime_error("SwiGLU NVIDIA: Unsupported data type");
    }
    
    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}
}
