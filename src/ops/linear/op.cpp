#include "op.hpp"
#include "../common.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.hpp"
#endif

namespace llaisys::ops {
namespace {

inline void validate_linear_tensors(const tensor_t& out,
                                    const tensor_t& in,
                                    const tensor_t& weight,
                                    const tensor_t& bias) {
    CHECK_SAME_DEVICE(out, in, weight, bias);

    ASSERT(out->ndim() == 2, "Linear: out must be 2D.");
    ASSERT(in->ndim() == 2, "Linear: input must be 2D.");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2D.");
    ASSERT(bias->ndim() == 1, "Linear: bias must be 1D.");

    const size_t N = in->shape()[0];
    const size_t I = in->shape()[1];
    const size_t O = weight->shape()[0];
    ASSERT(weight->shape()[1] == I, "Linear: weight shape mismatch with input features.");

    ASSERT(out->shape()[0] == N && out->shape()[1] == O, "Linear: output shape mismatch.");
    ASSERT(bias->shape()[0] == O, "Linear: bias shape mismatch.");

    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_SAME_DTYPE(out->dtype(), bias->dtype());

    ASSERT(llaisys::ops::common::allContiguous(in, weight, out), "Linear: out, input, and weight must be contiguous.");
    ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
}

template <typename T>
void linear_cpu(const tensor_t& out,
                const tensor_t& in,
                const tensor_t& weight,
                const tensor_t& bias) {
    const size_t N = in->shape()[0];
    const size_t I = in->shape()[1];
    const size_t O = weight->shape()[0];

    const T* x = reinterpret_cast<const T*>(in->data());
    const T* w = reinterpret_cast<const T*>(weight->data());
    const T* b = reinterpret_cast<const T*>(bias->data());
    T* y = reinterpret_cast<T*>(out->data());

    std::vector<float> bias_float(O);
    for (size_t o = 0; o < O; ++o) {
        bias_float[o] = llaisys::ops::common::to_float(b[o]);
    }

    for (size_t n = 0; n < N; ++n) {
        const T* x_row = x + n * I;
        for (size_t o = 0; o < O; ++o) {
            const T* w_row = w + o * I;
            float acc = bias_float[o];

            for (size_t i = 0; i < I; ++i) {
                acc += llaisys::ops::common::to_float(x_row[i]) * llaisys::ops::common::to_float(w_row[i]);
            }
            y[n * O + o] = llaisys::ops::common::from_float<T>(acc);
        }
    }
}

} // anonymous namespace

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    validate_linear_tensors(out, in, weight, bias);
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            linear_cpu<float>(out, in, weight, bias);
            return;
        case LLAISYS_DTYPE_F16:
            linear_cpu<llaisys::fp16_t>(out, in, weight, bias);
            return;
        case LLAISYS_DTYPE_BF16:
            linear_cpu<llaisys::bf16_t>(out, in, weight, bias);
            return;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
        break;
        
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::linear(out, in, weight, bias);
        return;
#endif
        
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops