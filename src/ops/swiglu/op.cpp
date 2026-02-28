#include "op.hpp"
#include "../common.hpp"
#include <cmath>

#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.hpp"
#endif

namespace llaisys::ops {
namespace {

inline void validate_swiglu_tensors(const tensor_t& out,
                                    const tensor_t& gate,
                                    const tensor_t& up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    ASSERT(out->ndim() == 2, "SwiGLU: out must be 2D.");
    ASSERT(gate->ndim() == 2, "SwiGLU: gate must be 2D.");
    ASSERT(up->ndim() == 2,   "SwiGLU: up must be 2D.");

    const size_t rows = out->shape()[0];
    const size_t cols = out->shape()[1];

    ASSERT(gate->shape()[0] == rows && gate->shape()[1] == cols, "SwiGLU: gate shape mismatch.");
    ASSERT(up->shape()[0] == rows   && up->shape()[1] == cols,   "SwiGLU: up shape mismatch.");

    ASSERT(llaisys::ops::common::allContiguous(out, gate, up), 
           "SwiGLU: all tensors must be contiguous.");
}

template <typename T>
void swiglu_cpu(const tensor_t& out,
                const tensor_t& gate,
                const tensor_t& up) {
    // Flattern arrays
    const size_t N = out->shape()[0] * out->shape()[1];

    const T* gate_ptr = reinterpret_cast<const T*>(gate->data());
    const T* up_ptr   = reinterpret_cast<const T*>(up->data());
    T* out_ptr        = reinterpret_cast<T*>(out->data());

    for (size_t i = 0; i < N; ++i) {
        const float g = llaisys::ops::common::to_float(gate_ptr[i]);
        const float u = llaisys::ops::common::to_float(up_ptr[i]);

        // Formula: out = up * (gate / (1 + exp(-gate)))
        // This is equivalent to: out = up * SiLU(gate)
        
        // Calculate SiLU(g)
        // Optimization: 1 / (1 + exp(-g)) is Sigmoid(g)
        // So SiLU(g) = g * Sigmoid(g)
        const float silu = g / (1.0f + std::exp(-g));

        const float res = u * silu;

        out_ptr[i] = llaisys::ops::common::from_float<T>(res);
    }
}
} // anonymous namespace

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    validate_swiglu_tensors(out, gate, up);

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
            case LLAISYS_DTYPE_F32:
                swiglu_cpu<float>(out, gate, up);
                return;
            case LLAISYS_DTYPE_F16:
                swiglu_cpu<llaisys::fp16_t>(out, gate, up);
                return;
            case LLAISYS_DTYPE_BF16:
                swiglu_cpu<llaisys::bf16_t>(out, gate, up);
                return;
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        nvidia::swiglu(out, gate, up);
        return;
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops