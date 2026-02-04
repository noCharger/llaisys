#include "op.hpp"
#include "../common.hpp"

namespace llaisys::ops {
namespace {

inline void validate_rms_norm_tensors(const tensor_t& out,
                                      const tensor_t& in,
                                      const tensor_t& weight) {
    CHECK_SAME_DEVICE(out, in, weight);

    ASSERT(out->ndim() == 2, "RMSNorm: out must be 2D.");
    ASSERT(in->ndim() == 2, "RMSNorm: input must be 2D.");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1D.");

    const size_t N = in->shape()[0];
    const size_t D = in->shape()[1];

    ASSERT(out->shape()[0] == N && out->shape()[1] == D, "RMSNorm: output shape mismatch.");
    ASSERT(weight->shape()[0] == D, "RMSNorm: weight length must match input feature size.");

    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    ASSERT(llaisys::ops::common::allContiguous(in, weight, out), "RMSNorm: all tensors must be contiguous.");
}

template <typename T>
void rms_norm_cpu(const tensor_t& out,
                  const tensor_t& in,
                  const tensor_t& weight,
                  float eps) {
    const size_t N = in->shape()[0];
    const size_t D = in->shape()[1];

    const T* in_ptr = reinterpret_cast<const T*>(in->data());
    const T* w_ptr  = reinterpret_cast<const T*>(weight->data());
    T* out_ptr      = reinterpret_cast<T*>(out->data());

    for (size_t i = 0; i < N; ++i) {
        const T* in_row = in_ptr + i * D;

        float sumsq = 0.0f;
        for (size_t j = 0; j < D; ++j) {
            const float x = llaisys::ops::common::to_float(in_row[j]);
            sumsq += x * x;
        }

        const float mean = sumsq / static_cast<float>(D);
        const float inv_rms = 1.0f / std::sqrt(mean + eps);

        T* out_row = out_ptr + i * D;
        for (size_t j = 0; j < D; ++j) {
            const float x = llaisys::ops::common::to_float(in_row[j]);
            const float w = llaisys::ops::common::to_float(w_ptr[j]);
            const float y = x * inv_rms * w;
            out_row[j] = llaisys::ops::common::from_float<T>(y);
        }
    }
}

} // anonymous namespace

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    validate_rms_norm_tensors(out, in, weight);

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
            case LLAISYS_DTYPE_F32:
                rms_norm_cpu<float>(out, in, weight, eps);
                return;
            case LLAISYS_DTYPE_F16:
                rms_norm_cpu<llaisys::fp16_t>(out, in, weight, eps);
                return;
            case LLAISYS_DTYPE_BF16:
                rms_norm_cpu<llaisys::bf16_t>(out, in, weight, eps);
                return;
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
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