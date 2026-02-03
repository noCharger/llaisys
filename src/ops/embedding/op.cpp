#include "op.hpp"
#include "../common.hpp"

namespace llaisys::ops {
namespace {

inline void validate_embedding_tensors(const tensor_t& out, const tensor_t& index, const tensor_t& weight) {
    CHECK_SAME_DEVICE(out, index, weight);

    ASSERT(index->ndim() == 1, "Embedding: index must be 1D.");
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2D.");
    ASSERT(out->ndim() == 2, "Embedding: out must be 2D.");

    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index dtype must be int64.");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    const size_t N = index->shape()[0];
    const size_t D = weight->shape()[1];
    ASSERT(out->shape()[0] == N && out->shape()[1] == D, "Embedding: output shape mismatch.");

    ASSERT(llaisys::ops::common::allContiguous(out, index, weight), "Embedding: all tensors must be contiguous.");
}

void embedding_cpu(const tensor_t& out, const tensor_t& index, const tensor_t& weight) {
    const size_t N = index->shape()[0];
    const size_t V = weight->shape()[0];
    const size_t D = weight->shape()[1];

    const int64_t* idx_ptr = reinterpret_cast<const int64_t*>(index->data());
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out->data());
    const uint8_t* w_ptr = reinterpret_cast<const uint8_t*>(weight->data());
    const size_t row_bytes = D * weight->elementSize();

    for (size_t i = 0; i < N; ++i) {
        const int64_t idx = idx_ptr[i];
        ASSERT(idx >= 0 && static_cast<size_t>(idx) < V, "Embedding: index out of range.");
        const size_t idx_u = static_cast<size_t>(idx);
        void* dst = static_cast<void*>(out_ptr + i * row_bytes);
        const void* src = static_cast<const void*>(w_ptr + idx_u * row_bytes);
        // Use runtime device-aware memcpy for CPU (H2H). This is equivalent to std::memcpy on CPU
        llaisys::core::context().runtime().api()->memcpy_sync(dst, src, row_bytes, LLAISYS_MEMCPY_H2H);
    }    
}

} // anonymous namespace

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    validate_embedding_tensors(out, index, weight);

    // CPU path
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        // Ensure CPU runtime is active before using runtime APIs
        llaisys::core::context().setDevice(LLAISYS_DEVICE_CPU, out->deviceId());
        embedding_cpu(out, index, weight);
        return;
    }

    // Non-CPU devices
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
