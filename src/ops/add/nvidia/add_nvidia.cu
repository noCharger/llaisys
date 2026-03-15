#include "add_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<<<blocksPerGrid, threadsPerBlock>>>((float *)c, (const float *)a, (const float *)b, size);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<<<blocksPerGrid, threadsPerBlock>>>((half *)c, (const half *)a, (const half *)b, size);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<<<blocksPerGrid, threadsPerBlock>>>((__nv_bfloat16 *)c, (const __nv_bfloat16 *)a, (const __nv_bfloat16 *)b, size);
        break;
    case LLAISYS_DTYPE_I32:
        add_kernel<<<blocksPerGrid, threadsPerBlock>>>((int *)c, (const int *)a, (const int *)b, size);
        break;
    default:
        throw std::runtime_error("Unsupported data type for add on NVIDIA device");
    }
    
    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

}
