#include "rearrange_nvidia.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace llaisys::ops::nvidia {

void rearrange(tensor_t out, tensor_t in) {
    size_t size_bytes = out->numel() * out->elementSize();
    cudaError_t err = cudaMemcpyAsync(out->data(), in->data(), size_bytes, cudaMemcpyDeviceToDevice, 0);
    if (err != cudaSuccess) {
        throw std::runtime_error("Rearrange NVIDIA: cudaMemcpyAsync failed");
    }
}

} // namespace llaisys::ops::nvidia
