#include "rms_norm_nvidia.hpp"
#include <cuda_runtime.h>
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

    float mean_sq = sdata[0] / D;
    float inv_rms = rsqrtf(mean_sq + eps);

    for (int i = tid; i < D; i += blockDim.x) {
        float val = static_cast<float>(in_row[i]);
        float w = static_cast<float>(weight[i]);
        out_row[i] = static_cast<T>(val * inv_rms * w);
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    int N = in->shape()[0];
    int D = in->shape()[1];
    int threads = 256;
    int blocks = N;
    size_t smem = threads * sizeof(float);

    if (out->dtype() == LLAISYS_DTYPE_F32) {
        rms_norm_kernel<float><<<blocks, threads, smem>>>((float*)out->data(), (const float*)in->data(), (const float*)weight->data(), eps, D);
    } else {
        throw std::runtime_error("RMSNorm NVIDIA: Only F32 supported");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("Kernel launch failed");
}
}
