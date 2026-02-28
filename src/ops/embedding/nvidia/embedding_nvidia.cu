#include "embedding_nvidia.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void embedding_kernel(T* out, const int64_t* index, const T* weight, int N, int D, int V) {
    int d_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (n_idx < N && d_idx < D) {
        int64_t vocab_idx = index[n_idx];
        if (vocab_idx >= 0 && vocab_idx < V) {
            out[n_idx * D + d_idx] = weight[vocab_idx * D + d_idx];
        }
    }
}

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    int N = index->numel();
    int D = weight->shape()[1];
    int V = weight->shape()[0];
    
    dim3 blockDim(32, 4);
    dim3 gridDim((D + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    if (out->dtype() == LLAISYS_DTYPE_F32) {
        embedding_kernel<float><<<gridDim, blockDim>>>((float*)out->data(), (const int64_t*)index->data(), (const float*)weight->data(), N, D, V);
    } else {
        throw std::runtime_error("Embedding NVIDIA: Only F32 supported");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("Kernel launch failed");
}
}
