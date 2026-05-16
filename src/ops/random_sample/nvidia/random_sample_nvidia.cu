#include "random_sample_nvidia.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

namespace llaisys::ops::nvidia {

#define CUDA_CHECK(x)                                                     \
    do {                                                                  \
        cudaError_t err__ = (x);                                          \
        if (err__ != cudaSuccess) {                                       \
            throw std::runtime_error(cudaGetErrorString(err__));          \
        }                                                                 \
    } while (0)

#define CURAND_CHECK(x)                                                   \
    do {                                                                  \
        curandStatus_t st__ = (x);                                        \
        if (st__ != CURAND_STATUS_SUCCESS) {                              \
            throw std::runtime_error("curand call failed");              \
        }                                                                 \
    } while (0)

template <typename T>
struct CastToFloatOp {
    __device__ __forceinline__ float operator()(const T& a) const {
        return static_cast<float>(a);
    }
};

template <>
struct CastToFloatOp<half> {
    __device__ __forceinline__ float operator()(const half& a) const {
        return __half2float(a);
    }
};

template <>
struct CastToFloatOp<__nv_bfloat16> {
    __device__ __forceinline__ float operator()(const __nv_bfloat16& a) const {
        return __bfloat162float(a);
    }
};

template <typename T>
__global__ void compute_scaled_probs_kernel(
    float* probs,
    int* indices,
    const T* logits,
    float max_scaled,
    float inv_temp,
    int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        float x = CastToFloatOp<T>()(logits[tid]);
        float scaled = x * inv_temp;
        probs[tid] = __expf(scaled - max_scaled);
        indices[tid] = tid;
    }
}

template <typename T>
struct ScaleCastOp {
    float inv_temp;
    __host__ __device__ explicit ScaleCastOp(float inv_t) : inv_temp(inv_t) {}
    __device__ __forceinline__ float operator()(const T& a) const {
        return CastToFloatOp<T>()(a) * inv_temp;
    }
};

__global__ void compute_limit_kernel(
    const float* cumsum,
    int size,
    int k,
    float top_p,
    float* out_limit)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float total_sum = cumsum[size - 1];
        float k_sum = cumsum[k - 1];
        float p_sum = total_sum * top_p;
        *out_limit = fminf(k_sum, p_sum);
    }
}

__global__ void compute_target_kernel(
    const float* limit,
    const float* u01,
    float* target)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float u = u01[0];
        // curandGenerateUniform 生成区间通常是 (0,1]
        // 为避免极端边界命中末尾，这里做轻微钳制
        u = fminf(u, 0.99999994f);
        u = fmaxf(u, 0.0f);
        target[0] = u * limit[0];
    }
}

__global__ void lower_bound_kernel(
    const float* cumsum,
    int n,
    float target,
    int* out_idx)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int left = 0;
        int right = n; // [left, right)
        while (left < right) {
            int mid = left + ((right - left) >> 1);
            if (cumsum[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if (left >= n) left = n - 1;
        *out_idx = left;
    }
}

__global__ void write_sampled_token_kernel(
    int64_t* out_token,
    const int* sorted_indices,
    const int* sampled_pos)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out_token[0] = static_cast<int64_t>(sorted_indices[sampled_pos[0]]);
    }
}

struct RandomSampleWorkspace {
    int capacity = 0;

    float* d_probs = nullptr;
    float* d_probs_sorted = nullptr;
    float* d_cumsum = nullptr;

    int* d_indices = nullptr;
    int* d_indices_sorted = nullptr;

    float* d_max_scaled = nullptr;
    float* d_limit = nullptr;
    float* d_u01 = nullptr;
    float* d_target = nullptr;

    int* d_sampled_pos = nullptr;

    void* d_reduce_temp = nullptr;
    size_t reduce_temp_bytes = 0;

    void* d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;

    void* d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;

    curandGenerator_t rng = nullptr;
    bool rng_inited = false;

    ~RandomSampleWorkspace() {
        release();
    }

    void release() {
        if (d_probs) cudaFree(d_probs);
        if (d_probs_sorted) cudaFree(d_probs_sorted);
        if (d_cumsum) cudaFree(d_cumsum);
        if (d_indices) cudaFree(d_indices);
        if (d_indices_sorted) cudaFree(d_indices_sorted);
        if (d_max_scaled) cudaFree(d_max_scaled);
        if (d_limit) cudaFree(d_limit);
        if (d_u01) cudaFree(d_u01);
        if (d_target) cudaFree(d_target);
        if (d_sampled_pos) cudaFree(d_sampled_pos);

        if (d_reduce_temp) cudaFree(d_reduce_temp);
        if (d_sort_temp) cudaFree(d_sort_temp);
        if (d_scan_temp) cudaFree(d_scan_temp);

        if (rng_inited) {
            curandDestroyGenerator(rng);
            rng_inited = false;
            rng = nullptr;
        }

        d_probs = nullptr;
        d_probs_sorted = nullptr;
        d_cumsum = nullptr;
        d_indices = nullptr;
        d_indices_sorted = nullptr;
        d_max_scaled = nullptr;
        d_limit = nullptr;
        d_u01 = nullptr;
        d_target = nullptr;
        d_sampled_pos = nullptr;
        d_reduce_temp = nullptr;
        d_sort_temp = nullptr;
        d_scan_temp = nullptr;
        reduce_temp_bytes = 0;
        sort_temp_bytes = 0;
        scan_temp_bytes = 0;
        capacity = 0;
    }

    template <typename T>
    void ensure_capacity(int size, const T* logits_ptr, cudaStream_t stream) {
        if (size <= capacity) {
            return;
        }

        release();

        capacity = size;

        CUDA_CHECK(cudaMalloc(&d_probs, capacity * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_probs_sorted, capacity * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cumsum, capacity * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_indices, capacity * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_indices_sorted, capacity * sizeof(int)));

        CUDA_CHECK(cudaMalloc(&d_max_scaled, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_limit, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_u01, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_target, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sampled_pos, sizeof(int)));

        // 预探测 CUB workspace 大小
        {
            cub::TransformInputIterator<float, ScaleCastOp<T>, const T*>
                it(logits_ptr, ScaleCastOp<T>(1.0f));

            cub::DeviceReduce::Max(nullptr, reduce_temp_bytes, it, d_max_scaled, size);
            CUDA_CHECK(cudaMalloc(&d_reduce_temp, reduce_temp_bytes));
        }

        {
            cub::DeviceRadixSort::SortPairsDescending(
                nullptr, sort_temp_bytes,
                d_probs, d_probs_sorted,
                d_indices, d_indices_sorted,
                size);
            CUDA_CHECK(cudaMalloc(&d_sort_temp, sort_temp_bytes));
        }

        {
            cub::DeviceScan::InclusiveSum(
                nullptr, scan_temp_bytes,
                d_probs_sorted, d_cumsum, size);
            CUDA_CHECK(cudaMalloc(&d_scan_temp, scan_temp_bytes));
        }

        CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, 123456789ULL));
        CURAND_CHECK(curandSetStream(rng, stream));
        rng_inited = true;
    }
};

// 你可以把它挂到 op/context 上，做到真正复用
static RandomSampleWorkspace g_workspace;

template <typename T>
void random_sample_cub_optimized(
    tensor_t out_token,
    const T* logits_ptr,
    int size,
    float temp,
    float top_p,
    int top_k,
    cudaStream_t stream)
{
    if (size <= 0) {
        throw std::runtime_error("random_sample: size must be > 0");
    }

    if (!(top_p > 0.0f && top_p <= 1.0f)) {
        top_p = 1.0f;
    }

    if (!(temp > 0.0f)) {
        // 常见语义：temp<=0 退化成 greedy；这里简单按极小正数处理也可以
        temp = 1e-6f;
    }

    int k = (top_k > 0 && top_k < size) ? top_k : size;
    float inv_temp = 1.0f / temp;

    g_workspace.ensure_capacity(size, logits_ptr, stream);

    // 1) 先对“缩放后的 logits”求最大值，修复温度公式
    {
        cub::TransformInputIterator<float, ScaleCastOp<T>, const T*>
            scaled_it(logits_ptr, ScaleCastOp<T>(inv_temp));

        cub::DeviceReduce::Max(
            g_workspace.d_reduce_temp,
            g_workspace.reduce_temp_bytes,
            scaled_it,
            g_workspace.d_max_scaled,
            size,
            stream);
    }

    // 2) 计算 exp(logit/temp - max_scaled)
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    compute_scaled_probs_kernel<<<blocks, threads, 0, stream>>>(
        g_workspace.d_probs,
        g_workspace.d_indices,
        logits_ptr,
        /* max_scaled */ 0.0f,  // 占位，下面改成从 device 读不行，所以换一种写法
        inv_temp,
        size);

    // 上面 kernel 还不能直接读取 d_max_scaled 标量，所以这里用 cudaMemcpyAsync 拷一个 4B 标量到 host
    // 如果你追求完全 device-side，可改成把 d_max_scaled 作为指针传进 kernel。
    float h_max_scaled = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&h_max_scaled, g_workspace.d_max_scaled, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    compute_scaled_probs_kernel<<<blocks, threads, 0, stream>>>(
        g_workspace.d_probs,
        g_workspace.d_indices,
        logits_ptr,
        h_max_scaled,
        inv_temp,
        size);

    // 3) 排序
    cub::DeviceRadixSort::SortPairsDescending(
        g_workspace.d_sort_temp,
        g_workspace.sort_temp_bytes,
        g_workspace.d_probs,
        g_workspace.d_probs_sorted,
        g_workspace.d_indices,
        g_workspace.d_indices_sorted,
        size,
        0,
        sizeof(float) * 8,
        stream);

    // 4) 前缀和
    cub::DeviceScan::InclusiveSum(
        g_workspace.d_scan_temp,
        g_workspace.scan_temp_bytes,
        g_workspace.d_probs_sorted,
        g_workspace.d_cumsum,
        size,
        stream);

    // 5) GPU 上计算 limit
    compute_limit_kernel<<<1, 1, 0, stream>>>(
        g_workspace.d_cumsum, size, k, top_p, g_workspace.d_limit);

    // 6) GPU 上生成 [0,1] 随机数
    CURAND_CHECK(curandSetStream(g_workspace.rng, stream));
    CURAND_CHECK(curandGenerateUniform(g_workspace.rng, g_workspace.d_u01, 1));

    // 7) target = u * limit
    compute_target_kernel<<<1, 1, 0, stream>>>(
        g_workspace.d_limit, g_workspace.d_u01, g_workspace.d_target);

    // 8) lower_bound，只在前 k 个里找
    float h_target = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&h_target, g_workspace.d_target, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    lower_bound_kernel<<<1, 1, 0, stream>>>(
        g_workspace.d_cumsum, k, h_target, g_workspace.d_sampled_pos);

    // 9) 直接把最终 token 写到 device 输出
    write_sampled_token_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<int64_t*>(out_token->data()),
        g_workspace.d_indices_sorted,
        g_workspace.d_sampled_pos);

    CUDA_CHECK(cudaGetLastError());
}

void random_sample(tensor_t out_token, tensor_t logits, float temp, float top_p, int top_k) {
    int size = logits->numel();
    auto dtype = logits->dtype();

    // 如果你的 tensor/context 能拿到 stream，这里一定要接进去
    cudaStream_t stream = 0;

    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            random_sample_cub_optimized<float>(
                out_token, reinterpret_cast<const float*>(logits->data()),
                size, temp, top_p, top_k, stream);
            break;
        case LLAISYS_DTYPE_F16:
            random_sample_cub_optimized<half>(
                out_token, reinterpret_cast<const half*>(logits->data()),
                size, temp, top_p, top_k, stream);
            break;
        case LLAISYS_DTYPE_BF16:
            random_sample_cub_optimized<__nv_bfloat16>(
                out_token, reinterpret_cast<const __nv_bfloat16*>(logits->data()),
                size, temp, top_p, top_k, stream);
            break;
        default:
            throw std::runtime_error("RandomSample NVIDIA: unsupported dtype");
    }
}

} // namespace llaisys::ops::nvidia