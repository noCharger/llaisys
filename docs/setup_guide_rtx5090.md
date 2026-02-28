# LLAISYS 项目开发环境搭建与测试指南 (RTX 5090 适配版)

**适用环境配置：**
- **GPU**: NVIDIA RTX 5090 (Blackwell 架构)
- **CPU**: Intel Xeon Gold 6459C (16 Core)
- **RAM**: 90 GB
- **Driver**: 580.76.05
- **CUDA**: 12.8 (内置于 PyTorch 2.8.0 镜像)
- **Image**: PyTorch 2.8.0 / Python 3.12 (Ubuntu 22.04) / CUDA 12.8

---

## 1. 环境准备与代码获取

### 1.1 设置工作目录
**注意**：系统盘仅 30GB，务必在数据盘进行操作。

```bash
# 假设数据盘挂载在 /root/autodl-tmp 或类似目录
cd /root/autodl-tmp 
mkdir -p workspace
cd workspace
```

### 1.2 配置环境变量 (防止系统盘爆满)
HuggingFace 默认将模型缓存在系统盘 (`/root/.cache`)，这极易导致 30GB 系统盘空间耗尽。务必通过设置 `HF_HOME` 环境变量将缓存路径迁移到数据盘。

建议将以下内容添加到 `~/.bashrc`：

```bash
# Xmake 缓存路径
export XMAKE_GLOBALDIR="/root/autodl-tmp/.xmake"
# HuggingFace 模型缓存 (关键: 迁移到数据盘)
export HF_HOME="/root/autodl-tmp/cache/huggingface"
# Pip 缓存
export PIP_CACHE_DIR="/root/autodl-tmp/cache/pip"
```
执行 `source ~/.bashrc` 生效。

或者在 Python 代码中临时设置（不推荐，建议全局设置）：
```python
import os
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/huggingface'
```

### 1.3 网络加速 (学术资源加速)
AutoDL 提供了学术资源加速服务，可显著提升 GitHub 和 HuggingFace 的访问速度。

**终端启用加速**：
```bash
source /etc/network_turbo
```

**验证加速是否生效**：
```bash
env | grep proxy
# 应输出 http_proxy 和 https_proxy 环境变量
```

**取消加速**（如遇到网络连接问题）：
```bash
unset http_proxy && unset https_proxy
```

### 1.4 验证预装环境
检查当前镜像的 CUDA 和 PyTorch 版本是否符合预期：

```bash
# 验证 CUDA 版本
ldconfig -p | grep cuda
# 预期输出应包含 libcudart.so.12.x 或类似条目

# 验证 PyTorch 版本
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, CuDNN: {torch.backends.cudnn.version()}')"
```

### 1.5 克隆代码库
```bash
# 启用学术加速后执行
git clone -b feature/cuda-support https://github.com/noCharger/llaisys.git
cd llaisys
```

### 1.6 下载测试模型
本项目使用 `DeepSeek-R1-Distill-Qwen-1.5B` 作为测试模型。建议使用 `huggingface-cli` 下载到数据盘，以避免占用系统盘空间。

```bash
# 确保已安装 huggingface_hub
pip install huggingface_hub

# 设置模型下载路径 (数据盘)
export MODEL_PATH="/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B"
mkdir -p $MODEL_PATH

# 方式一：使用 HuggingFace 镜像站 (推荐，需先取消学术加速)
unset http_proxy && unset https_proxy
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir $MODEL_PATH

# 方式二：使用学术加速 (备选)
# source /etc/network_turbo
# huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir $MODEL_PATH
```

---

## 2. 依赖安装

### 2.1 系统级依赖 (Xmake & Build Tools)
```bash
# 安装 Xmake (使用国内镜像加速)
export XMAKE_MIRROR_URL="https://gitee.com/tboox/xmake"
curl -fsSL https://xmake.io/shget.text | bash

# 刷新环境变量
source ~/.xmake/profile

# 验证 Xmake 安装
xmake --version

# 安装基础编译工具 (AutoDL 镜像通常已预装，可跳过或作为检查步骤)
# apt-get update && apt-get install -y build-essential cmake git
```

### 2.2 Python 依赖
当前镜像已预装 PyTorch 2.8.0 (Python 3.12 / CUDA 12.8)，因此**无需重新创建 Conda 环境或安装 PyTorch**。只需安装项目其余依赖：

```bash
# 确保使用系统预装的 Python 3.12
which python
# 预期输出: /usr/bin/python 或 conda 默认环境路径

# 安装项目依赖 (跳过 torch)
pip install transformers pytest numpy
```

---

## 3. 编译与构建 (CUDA 模式)

### 3.1 配置项目
针对 RTX 5090 (Blackwell)，我们需要启用 CUDA 支持。

```bash
# 配置构建：启用 CUDA，模式为 Release，详细输出
xmake f --nv-gpu=y -m release -v
```

**预期输出：**
- `checking for cuda SDK ... ok`
- `checking for nvcc ... /usr/local/cuda/bin/nvcc`

**常见错误排查：**
- **Error: cuda not found**: 检查 `nvcc --version`。如果未找到，需安装 CUDA Toolkit (推荐 12.4 或 12.6)。
- **Error: unsupported gpu architecture**: 检查 `xmake/nvidia.lua` 中的 `add_cugencodes`。RTX 5090 架构代号通常为 `sm_100` 或 `compute_100` (待确认，通常 `native` 选项会自动识别)。

### 3.2 编译后端
```bash
xmake
```

**预期输出：**
- `[100%]: build ok!`
- 生成 `libllaisys.so` (Linux) 或 `libllaisys.dylib` (macOS)

### 3.3 安装 Python 包
将编译好的 C++ 库安装到 Python 环境中：

```bash
# 安装 llaisys 库到当前 python 环境
xmake install

# 或者手动安装 python 包
pip install ./python/ --force-reinstall
```

---

## 4. 运行验证 (项目 #2 CUDA 测试)

### 4.1 验证 Runtime API
首先验证 CUDA Runtime API 是否正确实现及设备识别。

```bash
python test/test_runtime.py --device nvidia
```

**预期输出：**
```
...
test_device_count ... ok
test_memory_allocation ... ok
...
Ran X tests in X.XXXs
OK
```

### 4.2 验证 CUDA 算子
逐个运行算子测试，确保正确性。

```bash
# 批量运行所有算子测试 (快速验证)
python test/test_ops.py --device nvidia
```

如果遇到错误，可以单独运行特定算子的测试（例如 `python test/ops/linear.py --device nvidia`）。

**预期输出：** 所有测试均显示 `OK`。

### 4.3 验证端到端推理 (Inference)
在算子测试通过后，运行一次基础的推理测试（不启用 benchmark 循环）。

```bash
python test/test_infer.py --model $MODEL_PATH --device nvidia --test
```

**预期输出：** 生成正确的文本，并显示 `Test passed!`。

### 4.4 端到端延迟基准测试 (Latency Benchmark)
确认功能正常后，使用 `--benchmark` 参数进行性能评估。

```bash
python test/test_infer.py --model $MODEL_PATH --device nvidia --benchmark --repeat 20
```

**输出示例：**
```text
=== Benchmarking ===
Warming up...
Iter 1: 120.50ms, 45.20 tokens/s
...
Device: nvidia
End-to-End Latency:
  Mean: 120.50 ms
  P50:  118.20 ms
  P90:  125.10 ms
  P99:  145.00 ms
Throughput:
  Mean: 45.20 tokens/s
```

---

## 5. 常见问题排查 (Troubleshooting)

### Q1: `CUDA error: no kernel image is available for execution on the device`
- **原因**：编译时的架构 (Compute Capability) 与当前 GPU 不匹配。
- **解决**：在 `xmake/nvidia.lua` 中，确保使用了 `add_cugencodes("native")`，或者手动指定架构（RTX 5090 可能是 sm_100，具体视发布情况而定）。对于较旧的 CUDA Toolkit，可能需要更新 Toolkit。

### Q2: `ImportError: libllaisys.so: cannot open shared object file`
- **原因**：Python 找不到 C++ 共享库。
- **解决**：
  1. 确保运行了 `xmake install`。
  2. 检查 `LD_LIBRARY_PATH` 是否包含库路径（通常 pip install 会自动处理，但在某些环境下可能需要手动设置）。
  3. 尝试 `export PYTHONPATH=$PYTHONPATH:$(pwd)/python` 在项目根目录下运行。

### Q3: 显存不足 (OOM)
- **原因**：测试用例申请了过大显存。
- **解决**：虽然 5090 显存很大，但如果有其他进程占用，仍可能 OOM。使用 `nvidia-smi` 检查显存占用。

### Q4: 精度误差 (Assertion Error)
- **原因**：CUDA 浮点计算与 CPU/PyTorch 存在微小差异。
- **解决**：测试脚本通常有 `atol` (绝对误差) 和 `rtol` (相对误差) 阈值。如果是非常小的差异（如 1e-6 级别），通常是正常的。如果差异很大，检查 CUDA 核函数逻辑。

---

## 6. 性能验证 (可选)

为了确认 RTX 5090 的性能优势，可以运行 benchmark 模式（需自行编写或使用现有脚本的 profile 功能）：

```bash
# 例如，如果测试脚本支持 --profile
python test/ops/linear.py --device nvidia --profile
```

对比 CPU 和 GPU 的运行时间，应该能看到显著的加速（通常 10x - 100x）。
