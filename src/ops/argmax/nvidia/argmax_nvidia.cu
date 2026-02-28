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
    float local_max_val = -FLT_MAX;
    int local_max_idx = -1;
    
    for (int i = tid; i < size; i += blockDim.x) {
        float val = static_cast<float>(in[i]);
        if (val > local_max_val) {
            local_max_val = val;
            local_max_idx = i;
        }
    }
    
    __shared__ float s_val[1024];
    __shared__ int s_idx[1024];
    
    s_val[tid] = local_max_val;
    s_idx[tid] = local_max_idx;
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
    int threads = 1024;
    
    if (in->dtype() == LLAISYS_DTYPE_F32) {
        argmax_kernel_single_block<float><<<1, threads>>>((float*)max_val->data(), (int64_t*)max_idx->data(), (const float*)in->data(), size);
    } else if (in->dtype() == LLAISYS_DTYPE_F16) {
        argmax_kernel_single_block<half><<<1, threads>>>((half*)max_val->data(), (int64_t*)max_idx->data(), (const half*)in->data(), size);
    } else if (in->dtype() == LLAISYS_DTYPE_BF16) {
        argmax_kernel_single_block<__nv_bfloat16><<<1, threads>>>((__nv_bfloat16*)max_val->data(), (int64_t*)max_idx->data(), (const __nv_bfloat16*)in->data(), size);
    } else {
        throw std::runtime_error("Argmax NVIDIA: Unsupported data type");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("Kernel launch failed");
}
}
