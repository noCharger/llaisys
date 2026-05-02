#include "op.hpp"
#include "../common.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.hpp"
#endif

namespace llaisys::ops {
namespace {

template<typename T>
std::pair<int64_t, T> compute_argmax(const T* data, size_t n) {
    if (n == 0) return {0, T{0}};

    float best_val_f = llaisys::ops::common::to_float(data[0]);
    int64_t best_idx = 0;

    const T* current = data + 1;
    for (size_t i = 1; i < n; ++i, ++current) {
        float cur_f = llaisys::ops::common::to_float(*current);
        if (cur_f > best_val_f) {
            best_val_f = cur_f;
            best_idx = static_cast<int64_t>(i);
        }
    }

    return {best_idx, llaisys::ops::common::from_float<T>(best_val_f)};
}

template<typename T>
void argmax_cpu(tensor_t max_idx, tensor_t max_val, const tensor_t& vals) {
    const T* data = reinterpret_cast<const T*>(vals->data());
    const size_t n = vals->numel();

    auto [best_idx, best_val] = compute_argmax<T>(data, n);

    *reinterpret_cast<int64_t*>(max_idx->data()) = best_idx;
    *reinterpret_cast<T*>(max_val->data()) = best_val;
}

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

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    
#ifdef ENABLE_NVIDIA_API
    if (vals->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        nvidia::argmax(max_val, max_idx, vals);
        return;
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}

void argmax_batch(tensor_t out_indices, tensor_t logits) {
    ASSERT(out_indices && logits, "ArgmaxBatch: tensors must not be null.");
    ASSERT(out_indices->ndim() == 1, "ArgmaxBatch: out_indices must be 1D [N].");
    ASSERT(logits->ndim() == 2, "ArgmaxBatch: logits must be 2D [N, voc].");
    ASSERT(out_indices->shape()[0] == logits->shape()[0],
           "ArgmaxBatch: row count of out_indices and logits must match.");
    ASSERT(out_indices->dtype() == LLAISYS_DTYPE_I64,
           "ArgmaxBatch: out_indices must be int64.");
    // Contiguity required by the slice→view→argmax dispatch below.
    ASSERT(logits->isContiguous(),
           "ArgmaxBatch: logits must be contiguous.");
    ASSERT(out_indices->isContiguous(),
           "ArgmaxBatch: out_indices must be contiguous.");

    const size_t n = logits->shape()[0];
    const size_t voc = logits->shape()[1];

    // single argmax wants a max_val output; reuse one scratch across rows.
    tensor_t max_val_scratch = Tensor::create(
        std::vector<size_t>{1}, logits->dtype(),
        logits->deviceType(), logits->deviceId());

    for (size_t i = 0; i < n; ++i) {
        tensor_t row_2d = logits->slice(0, i, i + 1);
        tensor_t row_1d = row_2d->view(std::vector<size_t>{voc});
        tensor_t idx_slice = out_indices->slice(0, i, i + 1);
        argmax(idx_slice, max_val_scratch, row_1d);
    }
}
} // namespace llaisys::ops