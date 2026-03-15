#include "linear_nvidia.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include "utils/cuda_check.cuh"

namespace llaisys::ops::nvidia {
using namespace llaisys::utils;

// Vectorized types helper
template <typename T>
struct VecTraits;

template <>
struct VecTraits<float> {
    using Type = float4;
    static constexpr int size = 4;
};

template <>
struct VecTraits<half> {
    using Type = float4;
    static constexpr int size = 8;
};

template <>
struct VecTraits<__nv_bfloat16> {
    using Type = float4;
    static constexpr int size = 8;
};

template <typename T>
__global__ void add_bias_kernel(T* out, const T* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if (idx < size) {
        out[idx] += bias[idx % cols];
    }
}

// Vectorized Bias Add Kernel
template <typename T>
__global__ void add_bias_kernel_vectorized(T* out, const T* bias, int rows, int cols) {
    using VecType = typename VecTraits<T>::Type;
    int vec_cols = cols / VecTraits<T>::size;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_vecs = rows * vec_cols;

    if (idx < total_vecs) {
        int row = idx / vec_cols;
        int vec_col = idx % vec_cols;
        int scalar_col_offset = vec_col * VecTraits<T>::size;

        VecType b_vec = reinterpret_cast<const VecType*>(bias)[vec_col];
        
        VecType* out_vec_ptr = reinterpret_cast<VecType*>(out);
        int out_vec_idx = row * vec_cols + vec_col;
        VecType o_vec = out_vec_ptr[out_vec_idx];

        if constexpr (std::is_same_v<T, float>) {
            o_vec.x += b_vec.x; o_vec.y += b_vec.y; o_vec.z += b_vec.z; o_vec.w += b_vec.w;
        } else if constexpr (std::is_same_v<T, half>) {
            __half2* o_h2 = reinterpret_cast<__half2*>(&o_vec);
            const __half2* b_h2 = reinterpret_cast<const __half2*>(&b_vec);
            #pragma unroll
            for(int i=0; i<4; ++i) o_h2[i] = __hadd2(o_h2[i], b_h2[i]);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            __nv_bfloat162* o_bf2 = reinterpret_cast<__nv_bfloat162*>(&o_vec);
            const __nv_bfloat162* b_bf2 = reinterpret_cast<const __nv_bfloat162*>(&b_vec);
            #pragma unroll
            for(int i=0; i<4; ++i) o_bf2[i] = __hadd2(o_bf2[i], b_bf2[i]);
        }

        out_vec_ptr[out_vec_idx] = o_vec;
    }
}


void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    int M = in->shape()[0];
    int K = in->shape()[1];
    int N = out->shape()[1];
    
    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaDataType_t AType, BType, CType;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;

    switch(out->dtype()) {
        case LLAISYS_DTYPE_F32: AType = BType = CType = CUDA_R_32F; break;
        case LLAISYS_DTYPE_F16: AType = BType = CType = CUDA_R_16F; break;
        case LLAISYS_DTYPE_BF16: AType = BType = CType = CUDA_R_16BF; break;
        default: throw std::runtime_error("Linear NVIDIA: Unsupported data type");
    }

    cublasStatus_t status = cublasGemmEx(handle, 
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
        bool aligned = (reinterpret_cast<uintptr_t>(out->data()) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(bias->data()) % 16 == 0);
        
        int vec_size = 0;
        if (out->dtype() == LLAISYS_DTYPE_F32) vec_size = 4;
        else vec_size = 8;

        if (aligned && (N % vec_size == 0)) {
            int vec_cols = N / vec_size;
            int total_vecs = M * vec_cols;
            int threads = 256;
            int blocks = (total_vecs + threads - 1) / threads;

            switch(out->dtype()) {
                case LLAISYS_DTYPE_F32:
                    add_bias_kernel_vectorized<float><<<blocks, threads>>>((float*)out->data(), (const float*)bias->data(), M, N);
                    break;
                case LLAISYS_DTYPE_F16:
                    add_bias_kernel_vectorized<half><<<blocks, threads>>>((half*)out->data(), (const half*)bias->data(), M, N);
                    break;
                case LLAISYS_DTYPE_BF16:
                    add_bias_kernel_vectorized<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out->data(), (const __nv_bfloat16*)bias->data(), M, N);
                    break;
            }
        } else {
            // Fallback to scalar kernel for unaligned pointers
            int size = M * N;
            int threads = 256;
            int blocks = (size + threads - 1) / threads;

            switch(out->dtype()) {
                case LLAISYS_DTYPE_F32:
                    add_bias_kernel<float><<<blocks, threads>>>((float*)out->data(), (const float*)bias->data(), M, N);
                    break;
                case LLAISYS_DTYPE_F16:
                    add_bias_kernel<half><<<blocks, threads>>>((half*)out->data(), (const half*)bias->data(), M, N);
                    break;
                case LLAISYS_DTYPE_BF16:
                    add_bias_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out->data(), (const __nv_bfloat16*)bias->data(), M, N);
                    break;
            }
        }
        
        if (cudaGetLastError() != cudaSuccess) {
            throw std::runtime_error("Bias kernel launch failed");
        }
    }
}

} // namespace llaisys::ops::nvidia