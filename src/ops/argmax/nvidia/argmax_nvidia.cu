#include "argmax_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <cfloat>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void argmax_kernel_single_block(T* out_val, int64_t* out_idx, const T* in, int size) {
    int tid = threadIdx.x;
    float max_val = -FLT_MAX;
    int max_idx = -1;
    
    for (int i = tid; i < size; i += blockDim.x) {
        float val = static_cast<float>(in[i]);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    
    // TODO: Use dynamic shared memory sized to blockDim.x instead of hardcoded 1024.
    __shared__ float s_val[1024];
    __shared__ int s_idx[1024];
    
    s_val[tid] = max_val;
    s_idx[tid] = max_idx;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *out_val = static_cast<T>(s_val[0]);
        *out_idx = (int64_t)s_idx[0];
    }
}

void argmax(tensor_t max_val, tensor_t max_idx, tensor_t in) {
    int size = in->numel();
    
    // TODO: Implement a staged multi-block reduction path for large tensors.
    constexpr int threads = 1024;
    
    auto dtype = in->dtype();
    
    switch(dtype) {
        case LLAISYS_DTYPE_F32:
            argmax_kernel_single_block<float><<<1, threads>>>((float*)max_val->data(), (int64_t*)max_idx->data(), (const float*)in->data(), size);
            break;
        case LLAISYS_DTYPE_F16:
            argmax_kernel_single_block<half><<<1, threads>>>((half*)max_val->data(), (int64_t*)max_idx->data(), (const half*)in->data(), size);
            break;
        case LLAISYS_DTYPE_BF16:
            argmax_kernel_single_block<__nv_bfloat16><<<1, threads>>>((__nv_bfloat16*)max_val->data(), (int64_t*)max_idx->data(), (const __nv_bfloat16*)in->data(), size);
            break;
        default:
            throw std::runtime_error("Argmax NVIDIA: Unsupported data type");
    }
    
    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}
}
