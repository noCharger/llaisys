#include "op.hpp"
#include "../common.hpp"
#include <cmath>

namespace llaisys::ops {
namespace {

inline void validate_rope_tensors(const tensor_t& out,
                                  const tensor_t& in,
                                  const tensor_t& pos_ids) {
    CHECK_SAME_DEVICE(out, in, pos_ids);

    ASSERT(out->ndim() == 3, "RoPE: out must be 3D [seqlen, nhead, d].");
    ASSERT(in->ndim() == 3, "RoPE: in must be 3D [seqlen, nhead, d].");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D [seqlen].");

    const size_t seqlen = in->shape()[0];
    const size_t nhead = in->shape()[1];
    const size_t d = in->shape()[2];

    ASSERT(out->shape()[0] == seqlen && out->shape()[1] == nhead && out->shape()[2] == d,
           "RoPE: output shape mismatch.");
    ASSERT(pos_ids->shape()[0] == seqlen, "RoPE: pos_ids length must match sequence length.");
    ASSERT(d % 2 == 0, "RoPE: head dimension d must be even.");

    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64.");

    ASSERT(llaisys::ops::common::allContiguous(in, pos_ids, out), 
           "RoPE: all tensors must be contiguous.");
}

template <typename T>
void rope_cpu(const tensor_t& out,
              const tensor_t& in,
              const tensor_t& pos_ids,
              float theta) {
    const size_t seqlen = in->shape()[0];
    const size_t nhead = in->shape()[1];
    const size_t d = in->shape()[2];
    const size_t half_d = d / 2;

    const T* in_ptr = reinterpret_cast<const T*>(in->data());
    const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids->data());
    T* out_ptr = reinterpret_cast<T*>(out->data());

    // Use double for theta constants to maintain precision
    const double theta_d = static_cast<double>(theta);
    const double d_d = static_cast<double>(d);

    for (size_t i = 0; i < seqlen; ++i) {
        const int64_t p_i = pos_ptr[i];

        for (size_t h = 0; h < nhead; ++h) {
            // Calculate offsets for contiguous memory access
            const size_t offset = (i * nhead + h) * d;
            const T* in_vec = in_ptr + offset;
            T* out_vec = out_ptr + offset;

            for (size_t j = 0; j < half_d; ++j) {
                // Calculate angle in double precision
                // phi = p_i / theta^(2j/d)
                const double exponent = 2.0 * static_cast<double>(j) / d_d;
                const double freq = 1.0 / std::pow(theta_d, exponent);
                const double angle = static_cast<double>(p_i) * freq;

                const double cos_val = std::cos(angle);
                const double sin_val = std::sin(angle);

                // Load inputs, casting to double for intermediate calculation
                const double a = static_cast<double>(llaisys::ops::common::to_float(in_vec[j]));
                const double b = static_cast<double>(llaisys::ops::common::to_float(in_vec[j + half_d]));

                // Apply rotation
                // a' = a * cos - b * sin
                // b' = b * cos + a * sin
                const double a_prime = a * cos_val - b * sin_val;
                const double b_prime = b * cos_val + a * sin_val;

                // Cast back to target type and store
                out_vec[j]          = llaisys::ops::common::from_float<T>(static_cast<float>(a_prime));
                out_vec[j + half_d] = llaisys::ops::common::from_float<T>(static_cast<float>(b_prime));
            }
        }
    }
}

} // anonymous namespace

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    validate_rope_tensors(out, in, pos_ids);

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
            case LLAISYS_DTYPE_F32:
                rope_cpu<float>(out, in, pos_ids, theta);
                return;
            case LLAISYS_DTYPE_F16:
                rope_cpu<llaisys::fp16_t>(out, in, pos_ids, theta);
                return;
            case LLAISYS_DTYPE_BF16:
                rope_cpu<llaisys::bf16_t>(out, in, pos_ids, theta);
                return;
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        TO_BE_IMPLEMENTED();
        return;
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops