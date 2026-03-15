#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>

namespace llaisys::utils {

inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        std::cerr << "[ERROR] CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << " \"" << cudaGetErrorString(result) << "\" " << func << std::endl;
        throw std::runtime_error("CUDA runtime error");
    }
}

inline void check_last_cuda_error(const char *const file, int const line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(err) << " \"" << cudaGetErrorString(err) << "\"" << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

// Helper to manage cuBLAS handle efficiently
inline cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }
    }
    return handle;
}

// Helper to manage temporary device memory efficiently
inline void* get_scratch_buffer(size_t size) {
    static thread_local void* ptr = nullptr;
    static thread_local size_t capacity = 0;

    if (size > capacity) {
        if (ptr) cudaFree(ptr);
        if (cudaMalloc(&ptr, size) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate scratch buffer");
        }
        capacity = size;
    }
    return ptr;
}

} // namespace llaisys::utils

#define CHECK_CUDA(val) llaisys::utils::check_cuda((val), #val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() llaisys::utils::check_last_cuda_error(__FILE__, __LINE__)
