#include "add_nvidia.hpp"
#include "utils/cuda_check.cuh"
#include "utils/check.hpp"
#include "utils/types.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace llaisys::ops::nvidia {

// Vectorized types helper
template <typename T>
struct VectorizedStorage;

template <>
struct VectorizedStorage<float> {
    using Type = float4;
    static constexpr int size = 4;
    static __device__ __forceinline__ Type add(Type a, Type b) {
        return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
    }
};

template <>
struct VectorizedStorage<half> {
    using Type = float4; // 8 x half
    static constexpr int size = 8;
    static __device__ __forceinline__ Type add(Type a, Type b) {
        Type c;
        auto* a_h2 = reinterpret_cast<const __half2*>(&a);
        auto* b_h2 = reinterpret_cast<const __half2*>(&b);
        auto* c_h2 = reinterpret_cast<__half2*>(&c);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            c_h2[i] = a_h2[i] + b_h2[i];
        }
        return c;
    }
};

template <>
struct VectorizedStorage<__nv_bfloat16> {
    using Type = float4; // 8 x bf16
    static constexpr int size = 8;
    static __device__ __forceinline__ Type add(Type a, Type b) {
        Type c;
        auto* a_bf2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        auto* b_bf2 = reinterpret_cast<const __nv_bfloat162*>(&b);
        auto* c_bf2 = reinterpret_cast<__nv_bfloat162*>(&c);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            c_bf2[i] = a_bf2[i] + b_bf2[i];
        }
        return c;
    }
};

template <>
struct VectorizedStorage<int> {
    using Type = int4;
    static constexpr int size = 4;
    static __device__ __forceinline__ Type add(Type a, Type b) {
        return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
    }
};

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
#if __CUDA_ARCH__ >= 800
             c[idx] = a[idx] + b[idx];
#else
             // Fallback for older architectures
             c[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
#endif
        } else {
             c[idx] = a[idx] + b[idx];
        }
    }
}

template <typename T>
__global__ void add_kernel_vectorized(T *c, const T *a, const T *b, size_t n_vec) {
    using VecType = typename VectorizedStorage<T>::Type;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_vec) {
        VecType v_a = reinterpret_cast<const VecType*>(a)[idx];
        VecType v_b = reinterpret_cast<const VecType*>(b)[idx];
        
        reinterpret_cast<VecType*>(c)[idx] = VectorizedStorage<T>::add(v_a, v_b);
    }
}

template <typename T>
void launch_add(T *c, const T *a, const T *b, size_t size) {
    // 16-byte alignment required for float4/int4
    bool aligned = (reinterpret_cast<uintptr_t>(a) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(b) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(c) % 16 == 0);

    if (aligned) {
        constexpr int vec_size = VectorizedStorage<T>::size;
        size_t n_vec = size / vec_size;
        
        if (n_vec > 0) {
            int threads = 256;
            int blocks = (n_vec + threads - 1) / threads;
            add_kernel_vectorized<<<blocks, threads>>>(c, a, b, n_vec);
            CHECK_LAST_CUDA_ERROR();
        }

        // Handle tail elements that don't fit in a vector
        size_t remainder = size % vec_size;
        if (remainder > 0) {
            size_t offset = n_vec * vec_size;
            add_kernel<<<1, 32>>>(c + offset, a + offset, b + offset, remainder);
            CHECK_LAST_CUDA_ERROR();
        }
    } else {
        // Fallback to scalar kernel for unaligned pointers
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        add_kernel<<<blocks, threads>>>(c, a, b, size);
        CHECK_LAST_CUDA_ERROR();
    }
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t size) {
    CHECK_ARGUMENT(c != nullptr, "Output buffer cannot be null");
    CHECK_ARGUMENT(a != nullptr, "Input buffer A cannot be null");
    CHECK_ARGUMENT(b != nullptr, "Input buffer B cannot be null");
    CHECK_ARGUMENT(size > 0, "Size must be positive");

    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_add((float *)c, (const float *)a, (const float *)b, size);
        break;
    case LLAISYS_DTYPE_F16:
        launch_add((half *)c, (const half *)a, (const half *)b, size);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_add((__nv_bfloat16 *)c, (const __nv_bfloat16 *)a, (const __nv_bfloat16 *)b, size);
        break;
    case LLAISYS_DTYPE_I32:
        launch_add((int *)c, (const int *)a, (const int *)b, size);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
