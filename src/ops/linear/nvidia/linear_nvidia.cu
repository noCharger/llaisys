#include "linear_nvidia.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <iostream>

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
    int size = rows * cols;
    if (idx < size) {
        out[idx] += bias[idx % cols];
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    int M = in->shape()[0];
    int K = in->shape()[1];
    int N = out->shape()[1];
    
    CublasHandle handle;
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaDataType_t AType, BType, CType;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;

    if (out->dtype() == LLAISYS_DTYPE_F32) {
        AType = BType = CType = CUDA_R_32F;
    } else if (out->dtype() == LLAISYS_DTYPE_F16) {
        AType = BType = CType = CUDA_R_16F;
    } else if (out->dtype() == LLAISYS_DTYPE_BF16) {
        AType = BType = CType = CUDA_R_16BF;
    } else {
        throw std::runtime_error("Linear NVIDIA: Unsupported data type");
    }

    cublasStatus_t status = cublasGemmEx(handle.get(), 
        CUBLAS_OP_T, CUBLAS_OP_N, 
        N, M, K, 
        &alpha,
        weight->data(), AType, K,
        in->data(), BType, K, 
        &beta, 
        out->data(), CType, N,
        computeType, CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS Error: " + std::to_string(status));
    }

    if (bias) {
        int size = M * N;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        if (out->dtype() == LLAISYS_DTYPE_F32) {
            add_bias_kernel<float><<<blocks, threads>>>((float*)out->data(), (const float*)bias->data(), M, N);
        } else if (out->dtype() == LLAISYS_DTYPE_F16) {
            add_bias_kernel<half><<<blocks, threads>>>((half*)out->data(), (const half*)bias->data(), M, N);
        } else if (out->dtype() == LLAISYS_DTYPE_BF16) {
            add_bias_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out->data(), (const __nv_bfloat16*)bias->data(), M, N);
        }
        
        if (cudaGetLastError() != cudaSuccess) {
            throw std::runtime_error("Bias kernel launch failed");
        }
    }
}

} // namespace llaisys::ops::nvidia