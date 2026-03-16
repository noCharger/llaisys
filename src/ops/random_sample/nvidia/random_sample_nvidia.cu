#include "random_sample_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <cfloat>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>

namespace llaisys::ops::nvidia {

__global__ void sample_kernel(
    int64_t* out_token,
    const float* cumsum,
    const int* indices_sorted,
    const float* rand_val_ptr,
    int size,
    float top_p,
    int top_k)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float total_sum = cumsum[size - 1];
        int k = (top_k > 0 && top_k < size) ? top_k : size;
        float pk = cumsum[k - 1];
        float pp = total_sum * top_p;
        float plimit = min(pk, pp);
        
        float rand_val = *rand_val_ptr;
        float target = rand_val * plimit;
        
        int left = 0;
        int right = k - 1;
        int idx = k - 1; 
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (cumsum[mid] >= target) {
                idx = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        if (idx >= size) idx = size - 1;
        
        *out_token = (int64_t)indices_sorted[idx];
    }
}

template <typename T>
__global__ void compute_probs_kernel(float* probs, int* indices, const T* logits, const float* max_val_ptr, float temp, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        float max_val = *max_val_ptr;
        float val = static_cast<float>(logits[tid]);
        float logit_diff = val - max_val;
        if (temp > 0.0f) {
            logit_diff /= temp;
        }
        probs[tid] = expf(logit_diff);
        indices[tid] = tid;
    }
}

template <typename T>
struct CastToFloatOp {
    __device__ __forceinline__ float operator()(const T& a) const {
        return static_cast<float>(a);
    }
};

static curandGenerator_t get_generator() {
    static curandGenerator_t gen;
    static bool initialized = false;
    if (!initialized) {
        if (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
            // Fallback or panic, but static init usually safe if cuda context exists
        }
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        initialized = true;
    }
    return gen;
}

template <typename T>
void random_sample_cub(tensor_t out_token, const T* logits_ptr, tensor_t workspace, int size, float temp, float top_p, int top_k) {
    char* ws_ptr = (char*)workspace->data();
    size_t elem_size = (workspace->dtype() == LLAISYS_DTYPE_F32) ? 4 : 
                       (workspace->dtype() == LLAISYS_DTYPE_I64 ? 8 : 
                       (workspace->dtype() == LLAISYS_DTYPE_F16 || workspace->dtype() == LLAISYS_DTYPE_BF16 ? 2 : 1));
    size_t ws_size = workspace->numel() * elem_size;
    
    auto align = [](size_t offset) {
        return (offset + 255) & ~255;
    };
    
    size_t offset = 0;
    
    float* d_max_out = (float*)(ws_ptr + offset);
    offset += align(sizeof(float));
    
    float* d_rand_val = (float*)(ws_ptr + offset);
    offset += align(sizeof(float));
    
    float* d_probs = (float*)(ws_ptr + offset);
    offset += align(size * sizeof(float));
    
    int* d_indices = (int*)(ws_ptr + offset);
    offset += align(size * sizeof(int));
    
    float* d_probs_sorted = (float*)(ws_ptr + offset);
    offset += align(size * sizeof(float));
    
    int* d_indices_sorted = (int*)(ws_ptr + offset);
    offset += align(size * sizeof(int));
    
    float* d_cumsum = (float*)(ws_ptr + offset);
    offset += align(size * sizeof(float));
    
    void* d_temp_storage = (void*)(ws_ptr + offset);
    
    if (offset > ws_size) throw std::runtime_error("Workspace too small for buffers");
    size_t temp_storage_bytes = ws_size - offset;
    
    cub::TransformInputIterator<float, CastToFloatOp<T>, const T*> d_logits_float(logits_ptr, CastToFloatOp<T>());
    
    size_t required_bytes = 0;
    cub::DeviceReduce::Max(nullptr, required_bytes, d_logits_float, d_max_out, size);
    if (required_bytes > temp_storage_bytes) throw std::runtime_error("Workspace too small for Reduce");
    
    cub::DeviceReduce::Max(d_temp_storage, required_bytes, d_logits_float, d_max_out, size);
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    compute_probs_kernel<<<blocks, threads>>>(d_probs, d_indices, logits_ptr, d_max_out, temp, size);
    
    required_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(nullptr, required_bytes, d_probs, d_probs_sorted, d_indices, d_indices_sorted, size);
    if (required_bytes > temp_storage_bytes) throw std::runtime_error("Workspace too small for Sort");
    
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, required_bytes, d_probs, d_probs_sorted, d_indices, d_indices_sorted, size);
    
    required_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, required_bytes, d_probs_sorted, d_cumsum, size);
    if (required_bytes > temp_storage_bytes) throw std::runtime_error("Workspace too small for Scan");
    
    cub::DeviceScan::InclusiveSum(d_temp_storage, required_bytes, d_probs_sorted, d_cumsum, size);
    
    curandGenerateUniform(get_generator(), d_rand_val, 1);
    
    sample_kernel<<<1, 1>>>((int64_t*)out_token->data(), d_cumsum, d_indices_sorted, d_rand_val, size, top_p, top_k);
    
    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

void random_sample(tensor_t out_token, tensor_t logits, tensor_t workspace, float temp, float top_p, int top_k) {
    int size = logits->numel();
    auto dtype = logits->dtype();
    
    switch(dtype) {
        case LLAISYS_DTYPE_F32:
            random_sample_cub<float>(out_token, (const float*)logits->data(), workspace, size, temp, top_p, top_k);
            break;
        case LLAISYS_DTYPE_F16:
            random_sample_cub<half>(out_token, (const half*)logits->data(), workspace, size, temp, top_p, top_k);
            break;
        case LLAISYS_DTYPE_BF16:
            random_sample_cub<__nv_bfloat16>(out_token, (const __nv_bfloat16*)logits->data(), workspace, size, temp, top_p, top_k);
            break;
        default:
            throw std::runtime_error("RandomSample NVIDIA: Unsupported data type");
    }
}

} // namespace llaisys::ops::nvidia
