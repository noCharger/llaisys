1. 构建系统适配

通过 --metax-gpu=y 开启沐曦 GPU 编译，配合 --use-mc=y 可切换两套 SDK 模式： xmake.lua:117-136

SDK 双模式的核心在 xmake/metax.lua 中：

默认 HT 模式：使用 htgpu_llvm/bin/htcc 编译器 + hcblas/hcdnn/hcruntime 库
MC 模式（--use-mc=y）：使用 mxgpu_llvm/bin/mxcc 编译器 + mcblas/mcdnn/mcruntime 库
SDK 根目录从环境变量 MACA_PATH / MACA_HOME / MACA_ROOT 读取
自定义 .maca 文件编译规则 metax.lua:1-57
API 兼容层 metax_ht2mc.h 通过大量 #define 宏将 HT API 名称映射到 MC API 名称，从而用同一套代码支持两个 SDK： metax_ht2mc.h:1-88

-- 沐曦
option("metax-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for MetaX GPU")
option_end()
 
option("use-mc")
    set_default(false)
    set_showmenu(true)
    set_description("Use MC version")
option_end()
 
if has_config("metax-gpu") then
    add_defines("ENABLE_METAX_API")
    if has_config("use-mc") then
        add_defines("ENABLE_METAX_MC_API")
    end
    includes("xmake/metax.lua")
end


2. 设备 Handle 适配
device::metax::Handle，负责管理 hcblas 和 hcdnn 句柄池，并读取设备属性（warpSize、线程块/Grid 大小）

Handle 的创建在 handle.cc 的设备路由表中注册： handle.cc:23-25 handle.cc:65-67

src/infiniop/devices/metax/metax_handle.h

```
#ifndef __INFINIOP_METAX_HANDLE_H__
#define __INFINIOP_METAX_HANDLE_H__
#include "../../handle.h"
#include <memory>
namespace device::metax {
struct Handle : public InfiniopHandle {
    Handle(int device_id);
    class Internal;
    auto internal() const -> const std::shared_ptr<Internal> &;
public:
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);
protected:
    Handle(infiniDevice_t device, int device_id);
private:
    std::shared_ptr<Internal> _internal;
};
} // namespace device::metax

```

src/infiniop/devices/metax/metax_handle.cc
```
#include "metax_common.h"
namespace device::metax {
Handle::Handle(infiniDevice_t device, int device_id)
    : InfiniopHandle{device, device_id},
      _internal(std::make_shared<Handle::Internal>(device_id)) {}
Handle::Handle(int device_id) : Handle(INFINI_DEVICE_METAX, device_id) {}
auto Handle::internal() const -> const std::shared_ptr<Internal> & {
    return _internal;
}
Handle::Internal::Internal(int device_id) {
    hcDeviceProp_t prop;
    hcGetDeviceProperties(&prop, device_id);
    _warp_size = prop.warpSize;
    _max_threads_per_block = prop.maxThreadsPerBlock;
    _block_size[0] = prop.maxThreadsDim[0];
    _block_size[1] = prop.maxThreadsDim[1];
    _block_size[2] = prop.maxThreadsDim[2];
    _grid_size[0] = prop.maxGridSize[0];
    _grid_size[1] = prop.maxGridSize[1];
    _grid_size[2] = prop.maxGridSize[2];
}
infiniStatus_t Handle::Internal::useMcblas(hcStream_t stream, const Fn<hcblasHandle_t> &f) const {
    auto handle = mcblas_handles.pop();
    if (!handle) {
        CHECK_MCBLAS(hcblasCreate(&(*handle)));
    }
    CHECK_MCBLAS(hcblasSetStream(*handle, stream));
    CHECK_STATUS(f(*handle));
    mcblas_handles.push(std::move(*handle));
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Handle::Internal::useMcdnn(hcStream_t stream, const Fn<hcdnnHandle_t> &f) const {
    auto handle = mcdnn_handles.pop();
    if (!handle) {
        CHECK_MCDNN(hcdnnCreate(&(*handle)));
    }
    CHECK_MCDNN(hcdnnSetStream(*handle, stream));
    CHECK_STATUS(f(*handle));
    mcdnn_handles.push(std::move(*handle));
    return INFINI_STATUS_SUCCESS;
}
int Handle::Internal::warpSize() const { return _warp_size; }
int Handle::Internal::maxThreadsPerBlock() const { return _max_threads_per_block; }
int Handle::Internal::blockSizeX() const { return _block_size[0]; }
int Handle::Internal::blockSizeY() const { return _block_size[1]; }
int Handle::Internal::blockSizeZ() const { return _block_size[2]; }
int Handle::Internal::gridSizeX() const { return _grid_size[0]; }
int Handle::Internal::gridSizeY() const { return _grid_size[1]; }
int Handle::Internal::gridSizeZ() const { return _grid_size[2]; }
hcdnnDataType_t getHcdnnDtype(infiniDtype_t dt) {
    switch (dt) {
    case INFINI_DTYPE_F16:
        return HCDNN_DATA_HALF;
    case INFINI_DTYPE_F32:
        return HCDNN_DATA_FLOAT;
    case INFINI_DTYPE_F64:
        return HCDNN_DATA_DOUBLE;
    case INFINI_DTYPE_BF16:
        return HCDNN_DATA_BFLOAT16;
    case INFINI_DTYPE_I8:
        return HCDNN_DATA_INT8;
    case INFINI_DTYPE_I32:
        return HCDNN_DATA_INT32;
    case INFINI_DTYPE_I64:
        return HCDNN_DATA_INT64;
    case INFINI_DTYPE_U8:
        return HCDNN_DATA_UINT8;
    default:
        return HCDNN_DATA_FLOAT;
    }
}
infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id) {
    *handle_ptr = new Handle(INFINI_DEVICE_METAX, device_id);
    return INFINI_STATUS_SUCCESS;
}
} // namespace device::metax


```

算子适配（InfiniOP）
沐曦 GPU 原生 .maca kernel 实现的算子（位于各 ops/*/metax/ 目录）

Sample code: 

add_metax.h

```
#ifndef __ADD_METAX_API_H__
#define __ADD_METAX_API_H__

#include "../../../elementwise/metax/elementwise_metax_api.h"

ELEMENTWISE_DESCRIPTOR(add, metax)

#endif // __ADD_METAX_API_H__
```

```
#include "../../../devices/metax/metax_common.h"
#include "rope_metax.h"

#include "../../../devices/metax/metax_kernel_common.h"

#include "../cuda/kernel.cuh"
```