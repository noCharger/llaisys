#include "add_nvidia.hpp"
#include <cuda_runtime.h>
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
    case LLAISYS_DTYPE_I32:
        add_kernel<<<blocksPerGrid, threadsPerBlock>>>((int *)c, (const int *)a, (const int *)b, size);
        break;
    default:
        throw std::runtime_error("Unsupported data type for add on NVIDIA device");
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

} // namespace llaisys::ops::nvidia
