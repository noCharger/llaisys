#include "op.hpp"
#include "../common.hpp"
#include "../../../include/llaisys.h"
#include "../../utils/check.hpp"
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>

#ifdef ENABLE_NVIDIA_API
#include "nvidia/random_sample_nvidia.hpp"
#endif

namespace llaisys::ops {

namespace {

template <typename Tidx, typename Tval>
struct KVPair {
    Tidx idx;
    Tval val;

    bool operator<(const KVPair &other) const {
        return val > other.val;
    }
};

template <typename T>
void random_sample_cpu(tensor_t out_token, tensor_t logits, float temp, float top_p, int top_k) {
    const T* logits_ptr = reinterpret_cast<const T*>(logits->data());
    int64_t* out_ptr = reinterpret_cast<int64_t*>(out_token->data());
    size_t n = logits->numel();

    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float random_val = dis(gen);

    using ComputeType = float;
    
    std::vector<KVPair<int64_t, ComputeType>> pairs(n);
    for (size_t i = 0; i < n; i++) {
        pairs[i] = {static_cast<int64_t>(i), common::to_float(logits_ptr[i])};
    }
    
    std::sort(pairs.begin(), pairs.end());

    auto const max_val = pairs[0].val;
    pairs[0].val = 1.0f;
    
    for (size_t i = 1; i < n; i++) {
        float current_val = pairs[i].val;
        pairs[i].val = pairs[i - 1].val + std::exp((current_val - max_val) / temp);
    }

    size_t k_idx = std::min(static_cast<size_t>(top_k), n);
    if (k_idx == 0) k_idx = n;
    
    auto const pk = pairs[k_idx - 1].val;
    auto const pp = pairs[n - 1].val * top_p;
    auto const plimit = random_val * std::min(pk, pp);

    *out_ptr = pairs.empty() ? 0 : pairs.back().idx;
    
    for (size_t i = 0; i < n; i++) {
        if (plimit <= pairs[i].val) {
            *out_ptr = pairs[i].idx;
            break;
        }
    }
}

void validate_random_sample_tensors(const tensor_t& out_token, const tensor_t& logits) {
    CHECK_SAME_DEVICE(out_token, logits);
    
    ASSERT(out_token->ndim() == 1 && out_token->shape()[0] == 1, 
           "RandomSample: out_token must be 1D with single element.");
    ASSERT(logits->ndim() == 1 || (logits->ndim() == 2 && logits->shape()[0] == 1), 
           "RandomSample: logits must be 1D or [1, voc].");
    
    ASSERT(out_token->dtype() == LLAISYS_DTYPE_I64, 
           "RandomSample: out_token tensor must be int64.");
}

} // namespace

void random_sample(tensor_t out_token, tensor_t logits, float temp, float top_p, int top_k) {
    validate_random_sample_tensors(out_token, logits);

    if (logits->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (logits->dtype()) {
            case LLAISYS_DTYPE_F32:
                random_sample_cpu<float>(out_token, logits, temp, top_p, top_k);
                return;
            case LLAISYS_DTYPE_F16:
                random_sample_cpu<llaisys::fp16_t>(out_token, logits, temp, top_p, top_k);
                return;
            case LLAISYS_DTYPE_BF16:
                random_sample_cpu<llaisys::bf16_t>(out_token, logits, temp, top_p, top_k);
                return;
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(logits->dtype());
        }
    }

    llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());
    
#ifdef ENABLE_NVIDIA_API
    if (logits->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        nvidia::random_sample(out_token, logits, temp, top_p, top_k);
        return;
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
