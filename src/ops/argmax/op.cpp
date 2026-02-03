#include "op.hpp"
#include "../common.hpp"

namespace llaisys::ops {
namespace {

// Direct conversion helpers to avoid the generic cast overhead
template<typename T>
inline float to_float_for_comparison(T val);

template<>
inline float to_float_for_comparison<float>(float val) {
    return val;  // No conversion needed
}

template<>
inline float to_float_for_comparison<llaisys::fp16_t>(llaisys::fp16_t val) {
    return llaisys::utils::_f16_to_f32(val);
}

template<>
inline float to_float_for_comparison<llaisys::bf16_t>(llaisys::bf16_t val) {
    return llaisys::utils::_bf16_to_f32(val);
}

template<typename T>
inline T from_float(float val);

template<>
inline float from_float<float>(float val) {
    return val;
}

template<>
inline llaisys::fp16_t from_float<llaisys::fp16_t>(float val) {
    return llaisys::utils::_f32_to_f16(val);
}

template<>
inline llaisys::bf16_t from_float<llaisys::bf16_t>(float val) {
    return llaisys::utils::_f32_to_bf16(val);
}

template<typename T>
std::pair<int64_t, T> compute_argmax(const T* data, size_t n) {
    if (n == 0) return {0, T{0}};
    
    float best_val_f = to_float_for_comparison(data[0]);
    int64_t best_idx = 0;
    
    // Use pointer arithmetic for potential auto-vectorization
    const T* current = data + 1;
    for (size_t i = 1; i < n; ++i, ++current) {
        float cur_f = to_float_for_comparison(*current);
        if (cur_f > best_val_f) {
            best_val_f = cur_f;
            best_idx = static_cast<int64_t>(i);
        }
    }
    
    return {best_idx, from_float<T>(best_val_f)};
}

// CPU implementation using direct conversions
template<typename T>
void argmax_cpu(tensor_t max_idx, tensor_t max_val, const tensor_t& vals) {
    const T* data = reinterpret_cast<const T*>(vals->data());
    const size_t n = vals->numel();
    
    auto [best_idx, best_val] = compute_argmax<T>(data, n);
    
    *reinterpret_cast<int64_t*>(max_idx->data()) = best_idx;
    *reinterpret_cast<T*>(max_val->data()) = best_val;
}

// Validation function
inline void validate_argmax_tensors(const tensor_t& max_idx, 
                                   const tensor_t& max_val, 
                                   const tensor_t& vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    
    ASSERT(max_idx->ndim() == 1 && max_idx->shape()[0] == 1, 
           "Argmax: max_idx must be 1D with single element.");
    ASSERT(max_val->ndim() == 1 && max_val->shape()[0] == 1, 
           "Argmax: max_val must be 1D with single element.");
    ASSERT(vals->ndim() == 1, "Argmax: vals must be 1D.");
    
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, 
           "Argmax: index tensor must be int64.");
    ASSERT(llaisys::ops::common::allContiguous(max_idx, max_val, vals), 
           "Argmax: all tensors must be contiguous.");
    ASSERT(vals->numel() > 0, "Argmax: vals must have at least one element.");
}

} // anonymous namespace

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    validate_argmax_tensors(max_idx, max_val, vals);
    
    // CPU path optimized with direct conversions
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (vals->dtype()) {
            case LLAISYS_DTYPE_F32:
                argmax_cpu<float>(max_idx, max_val, vals);
                return;
            case LLAISYS_DTYPE_F16:
                argmax_cpu<llaisys::fp16_t>(max_idx, max_val, vals);
                return;
            case LLAISYS_DTYPE_BF16:
                argmax_cpu<llaisys::bf16_t>(max_idx, max_val, vals);
                return;
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
        }
    }
    
    // Non-CPU devices
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    
    switch (vals->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops