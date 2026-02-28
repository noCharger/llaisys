#include "op.hpp"
#include "../../core/context/context.hpp"
#include "../../utils.hpp"
#include <cstring>

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rearrange_nvidia.hpp"
#endif

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        size_t size = out->numel() * out->elementSize();
        std::memcpy(out->data(), in->data(), size);
        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        nvidia::rearrange(out, in);
        return;
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
