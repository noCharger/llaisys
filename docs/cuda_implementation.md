# CUDA Implementation Details

This document describes the implementation of CUDA support for LLAISYS.

## 1. Build System

- **xmake/nvidia.lua**: Added to handle CUDA compilation.
  - Targets: `llaisys-device-nvidia`, `llaisys-ops-nvidia`.
  - Uses `add_rules("cuda")` and sets C++17 standard.
- **xmake.lua**:
  - Added `option("nv-gpu")` to enable CUDA support.
  - Conditionally includes `xmake/nvidia.lua` and adds dependencies when `nv-gpu` is enabled.
  - Added `add_includedirs("include")` to fix include path issues.
- **xmake/cpu.lua**:
  - Added `set_languages("cxx17")` to `llaisys-ops-cpu` target to fix compilation errors with `std::byte`.

## 2. Runtime API

- **src/device/nvidia/nvidia_runtime_api.cu**:
  - Implemented all `RuntimeAPI` functions using CUDA Runtime API.
  - Added `CHECK_CUDA` macro for error handling.
  - Mapped `llaisysMemcpyKind_t` to `cudaMemcpyKind`.
- **src/device/nvidia/nvidia_resource.cu**:
  - Implemented `Resource` class constructor and destructor.

## 3. Operators

- **Add Operator**:
  - **src/ops/add/nvidia/add_nvidia.cu**: Implemented `add_kernel` and `add` dispatch function.
  - **src/ops/add/nvidia/add_nvidia.hpp**: Header for the CUDA implementation.
  - **src/ops/add/op.cpp**: Updated to dispatch to NVIDIA implementation when device type is `LLAISYS_DEVICE_NVIDIA`.

## 4. Usage

To build with CUDA support:
```bash
xmake f --nv-gpu=y
xmake
```

To run tests:
```bash
python test/test_runtime.py --device nvidia
python test/ops/add.py --device nvidia
```
