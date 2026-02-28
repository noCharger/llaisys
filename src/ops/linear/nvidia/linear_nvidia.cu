#include "linear_nvidia.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>

namespace llaisys::ops::nvidia {

class CublasHandle {
public:
    CublasHandle() { cublasCreate(&handle_); }
    ~CublasHandle() { cublasDestroy(handle_); }
    cublasHandle_t get() { return handle_; }
private:
    cublasHandle_t handle_;
};

template <typename T>
__global__ void add_bias_kernel(T* out, const T* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        out[idx] = out[idx] + bias[idx % cols];
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    int M = in->shape()[0];
    int K = in->shape()[1];
    int N = out->shape()[1];
    
    CublasHandle handle;
    float alpha = 1.0f, beta = 0.0f;
    half alpha_h = __float2half(1.0f);
    half beta_h = __float2half(0.0f);

    if (out->dtype() == LLAISYS_DTYPE_F32) {
        cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
            (const float*)weight->data(), K,
            (const float*)in->data(), K, &beta, (float*)out->data(), N);
    } else if (out->dtype() == LLAISYS_DTYPE_F16) {
        cublasHgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha_h,
            (const half*)weight->data(), K,
            (const half*)in->data(), K, &beta_h, (half*)out->data(), N);
    } else if (out->dtype() == LLAISYS_DTYPE_BF16) {
        cublasGemmEx(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
            weight->data(), CUDA_R_16BF, K,
            in->data(), CUDA_R_16BF, K, &beta, out->data(), CUDA_R_16BF, N,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    } else {
        throw std::runtime_error("Linear NVIDIA: Unsupported data type");
    }

    if (bias) {
        int threads = 256;
        int blocks = (M * N + threads - 1) / threads;
        if (out->dtype() == LLAISYS_DTYPE_F32) {
            add_bias_kernel<float><<<blocks, threads>>>((float*)out->data(), (const float*)bias->data(), M, N);
        } else if (out->dtype() == LLAISYS_DTYPE_F16) {
            add_bias_kernel<half><<<blocks, threads>>>((half*)out->data(), (const half*)bias->data(), M, N);
        } else if (out->dtype() == LLAISYS_DTYPE_BF16) {
            add_bias_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out->data(), (const __nv_bfloat16*)bias->data(), M, N);
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("Kernel launch failed");
}
}
