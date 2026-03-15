#include "random_sample_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <cfloat>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void compute_probs_kernel(float* probs, int* indices, const T* logits, float max_val, float temp, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        float val = static_cast<float>(logits[tid]);
        if (temp > 0.0f && temp != 1.0f) {
            val /= temp;
        }
        probs[tid] = expf(val - max_val);
        indices[tid] = tid;
    }
}

template <typename T>
struct CastToFloatOp {
    __device__ __forceinline__ float operator()(const T& a) const {
        return static_cast<float>(a);
    }
};

template <typename T>
void random_sample_cub(tensor_t out_token, const T* logits_ptr, int size, float temp, float top_p, int top_k) {
    float* d_max_out;
    cudaMalloc(&d_max_out, sizeof(float));
    
    cub::TransformInputIterator<float, CastToFloatOp<T>, const T*> d_logits_float(logits_ptr, CastToFloatOp<T>());
    
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_logits_float, d_max_out, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_logits_float, d_max_out, size);
    
    float h_max_val;
    cudaMemcpy(&h_max_val, d_max_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_temp_storage);
    cudaFree(d_max_out);

    float* d_probs;
    int* d_indices;
    cudaMalloc(&d_probs, size * sizeof(float));
    cudaMalloc(&d_indices, size * sizeof(int));
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    compute_probs_kernel<<<blocks, threads>>>(d_probs, d_indices, logits_ptr, h_max_val, temp, size);

    float* d_probs_sorted;
    int* d_indices_sorted;
    cudaMalloc(&d_probs_sorted, size * sizeof(float));
    cudaMalloc(&d_indices_sorted, size * sizeof(int));
    
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_probs, d_probs_sorted, d_indices, d_indices_sorted, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_probs, d_probs_sorted, d_indices, d_indices_sorted, size);
    cudaFree(d_temp_storage);
    
    float* d_cumsum;
    cudaMalloc(&d_cumsum, size * sizeof(float));
    
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_probs_sorted, d_cumsum, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_probs_sorted, d_cumsum, size);
    cudaFree(d_temp_storage);

    float total_sum;
    cudaMemcpy(&total_sum, d_cumsum + size - 1, sizeof(float), cudaMemcpyDeviceToHost);
    
    int k = (top_k > 0 && top_k < size) ? top_k : size;
    float pk;
    cudaMemcpy(&pk, d_cumsum + k - 1, sizeof(float), cudaMemcpyDeviceToHost);
    
    float pp = total_sum * top_p;
    float plimit_val = std::min(pk, pp);
    
    float rand_val = (float)rand() / (float)RAND_MAX;
    float target_cumsum = rand_val * plimit_val;
    
    thrust::device_ptr<float> t_cumsum(d_cumsum);
    int sampled_idx = thrust::distance(t_cumsum, thrust::lower_bound(thrust::device, t_cumsum, t_cumsum + k, target_cumsum));
    
    if (sampled_idx >= size) sampled_idx = size - 1;
    
    int final_token;
    cudaMemcpy(&final_token, d_indices_sorted + sampled_idx, sizeof(int), cudaMemcpyDeviceToHost);
    
    int64_t final_token_i64 = final_token;
    cudaMemcpy(out_token->data(), &final_token_i64, sizeof(int64_t), cudaMemcpyHostToDevice);

    cudaFree(d_probs);
    cudaFree(d_indices);
    cudaFree(d_probs_sorted);
    cudaFree(d_indices_sorted);
    cudaFree(d_cumsum);
}

void random_sample(tensor_t out_token, tensor_t logits, float temp, float top_p, int top_k) {
    int size = logits->numel();
    auto dtype = logits->dtype();
    
    switch(dtype) {
        case LLAISYS_DTYPE_F32:
            random_sample_cub<float>(out_token, (const float*)logits->data(), size, temp, top_p, top_k);
            break;
        case LLAISYS_DTYPE_F16:
            random_sample_cub<half>(out_token, (const half*)logits->data(), size, temp, top_p, top_k);
            break;
        case LLAISYS_DTYPE_BF16:
            random_sample_cub<__nv_bfloat16>(out_token, (const __nv_bfloat16*)logits->data(), size, temp, top_p, top_k);
            break;
        default:
            throw std::runtime_error("RandomSample NVIDIA: Unsupported data type");
    }
    
    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed");
    }
}

} // namespace llaisys::ops::nvidia
