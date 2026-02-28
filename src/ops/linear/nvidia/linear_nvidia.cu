#include "linear_nvidia.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <iostream>

namespace llaisys::ops::nvidia {

// Helper to handle cuBLAS errors
#define CHECK_CUBLAS(status) \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("cuBLAS Error at line " + std::to_string(__LINE__)); \
    }

class CublasHandle {
public:
    CublasHandle() { cublasCreate(&handle_); }
    ~CublasHandle() { cublasDestroy(handle_); }
    cublasHandle_t get() { return handle_; }
private:
    cublasHandle_t handle_;
};

// Optimized Bias Kernel: Each thread handles one element
template <typename T>
__global__ void add_bias_kernel(T* out, const T* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if (idx < size) {
        // Since 'out' is Row-Major M x N:
        // row = idx / cols, col = idx % cols
        int col = idx % cols; 
        out[idx] = out[idx] + bias[col];
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // Shapes: In(M, K), Weight(N, K), Out(M, N)
    int M = in->shape()[0];
    int K = in->shape()[1];
    int N = out->shape()[1];
    
    CublasHandle handle;
    
    // We use float for alpha/beta even for FP16/BF16 
    // when using CUBLAS_COMPUTE_32F for higher precision accumulation.
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

    /*
      LOGIC RECAP:
      Row-major Out(M,N) = In(M,K) * Weight(N,K)^T
      In cuBLAS (Col-major):
      We want C(N,M) = Weight(N,K) * In^T(K,M)
      
      A = Weight. Memory is (N,K) row-major -> (K,N) col-major. 
          To get (N,K), use CUBLAS_OP_T. LDA = K.
      B = In. Memory is (M,K) row-major -> (K,M) col-major.
          This is exactly In^T(K,M). Use CUBLAS_OP_N. LDB = K.
      C = Out. Memory is (M,N) row-major -> (N,M) col-major. LDC = N.
    */

    CHECK_CUBLAS(cublasGemmEx(handle.get(), 
        CUBLAS_OP_T, CUBLAS_OP_N, 
        N, M, K, 
        &alpha,
        weight->data(), AType, K,
        in->data(), BType, K, 
        &beta, 
        out->data(), CType, N,
        computeType, CUBLAS_GEMM_DEFAULT));

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
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) throw std::runtime_error("Bias kernel launch failed");
    }

    // Synchronize only if you need to catch errors immediately (useful for your test script)
    cudaDeviceSynchronize();
}

} // namespace llaisys::ops::nvidia