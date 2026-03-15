#include "../runtime_api.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error: ") +          \
                                     cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

namespace llaisys::device::nvidia {

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device) {
    CHECK_CUDA(cudaSetDevice(device));
}

void deviceSynchronize() {
    CHECK_CUDA(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    return (llaisysStream_t)stream;
}

void destroyStream(llaisysStream_t stream) {
    CHECK_CUDA(cudaStreamDestroy((cudaStream_t)stream));
}

void streamSynchronize(llaisysStream_t stream) {
    CHECK_CUDA(cudaStreamSynchronize((cudaStream_t)stream));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    CHECK_CUDA(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    CHECK_CUDA(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    CHECK_CUDA(cudaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        cuda_kind = cudaMemcpyHostToHost;
        break;
    case LLAISYS_MEMCPY_H2D:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
    case LLAISYS_MEMCPY_D2H:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
    case LLAISYS_MEMCPY_D2D:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
    default:
        throw std::runtime_error("Unknown memcpy kind");
    }
    CHECK_CUDA(cudaMemcpy(dst, src, size, cuda_kind));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        cuda_kind = cudaMemcpyHostToHost;
        break;
    case LLAISYS_MEMCPY_H2D:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
    case LLAISYS_MEMCPY_D2H:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
    case LLAISYS_MEMCPY_D2D:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
    default:
        throw std::runtime_error("Unknown memcpy kind");
    }
    CHECK_CUDA(cudaMemcpyAsync(dst, src, size, cuda_kind, (cudaStream_t)stream));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
