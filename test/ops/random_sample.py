import sys
import os
import ctypes
import numpy as np
import argparse

import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

python_dir = os.path.abspath(os.path.join(parent_dir, "../python"))
sys.path.insert(0, python_dir)

import llaisys
from test_utils import random_tensor, check_equal, benchmark, zero_tensor, llaisys_dtype, llaisys_device

def create_llaisys_tensor(torch_tensor, device_name):
    """
    Creates a llaisys Tensor from a PyTorch tensor, handling device transfer.
    """
    shape = torch_tensor.shape
    dtype_name = "f32"
    if torch_tensor.dtype == torch.float16: dtype_name = "f16"
    elif torch_tensor.dtype == torch.bfloat16: dtype_name = "bf16"
    elif torch_tensor.dtype == torch.int64: dtype_name = "i64"
    
    llaisys_tensor = llaisys.Tensor(
        shape=shape,
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
        device_id=0,
    )
    
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    
    # Copy data from torch tensor to llaisys tensor
    kind = llaisys.MemcpyKind.H2D
    src_ptr = torch_tensor.data_ptr()
    
    if torch_tensor.device.type == "cuda":
        kind = llaisys.MemcpyKind.D2D
        
    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        src_ptr,
        bytes_,
        kind,
    )
    return llaisys_tensor

def create_workspace(device_name):
    """
    Allocates workspace memory for NVIDIA device sampling.
    """
    if device_name == "nvidia":
        ws_size = 32 * 1024 * 1024 # 32MB
        return llaisys.Tensor(
            shape=(ws_size,),
            dtype=llaisys_dtype("u8"),
            device=llaisys_device(device_name),
            device_id=0
        )
    return None

def get_output_token(out_token_llaisys, device_name):
    """
    Retrieves the sampled token from the llaisys output tensor.
    """
    if device_name == "nvidia":
        api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
        res = ctypes.c_int64()
        api.memcpy_sync(ctypes.byref(res), out_token_llaisys.data_ptr(), 8, llaisys.MemcpyKind.D2H)
        return res.value
    else:
        data_ptr = out_token_llaisys.data_ptr()
        return int(ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_int64)).contents.value)

def test_top_k(device):
    print("Testing Top-K constraint...")
    shape = (1, 1000)
    top_k = 10
    
    # Generate random logits
    logits, _ = random_tensor(shape, "f32", "cpu", scale=1.0)
    logits_torch = torch.from_numpy(logits) if isinstance(logits, np.ndarray) else logits
    
    # Determine allowed indices using PyTorch's topk
    vals, indices = torch.topk(logits_torch, top_k)
    allowed_indices = set(indices.flatten().tolist())
    
    # Prepare llaisys tensors
    logits_llaisys = create_llaisys_tensor(logits_torch, device)
    out_token, out_token_llaisys = zero_tensor((1,), "i64", device)
    workspace = create_workspace(device)

    # Run multiple iterations to verify constraint
    for _ in range(50):
        llaisys.Ops.random_sample(out_token_llaisys, logits_llaisys, workspace, 1.0, 1.0, top_k)
        token_id = get_output_token(out_token_llaisys, device)
        assert token_id in allowed_indices, f"Top-K failed: token {token_id} not in top {top_k} indices"
        
    print("Top-K passed.")

def test_top_p(device):
    print("Testing Top-P constraint...")
    shape = (1, 100)
    top_p = 0.5
    
    # Create a skewed distribution where the first element dominates or satisfies top_p
    # e.g., softmax([10.0, 8.0, ...]) -> p[0] is high
    logits_torch = torch.tensor([[10.0, 8.0, 5.0, 2.0] + [0.0]*96])
    
    logits_llaisys = create_llaisys_tensor(logits_torch, device)
    out_token, out_token_llaisys = zero_tensor((1,), "i64", device)
    workspace = create_workspace(device)
    
    # With this distribution and top_p=0.5, index 0 should be the only valid sample
    # (assuming index 0 prob > 0.5)
    for _ in range(50):
        llaisys.Ops.random_sample(out_token_llaisys, logits_llaisys, workspace, 1.0, top_p, 0)
        token_id = get_output_token(out_token_llaisys, device)
        assert token_id == 0, f"Top-P failed: token {token_id} selected, expected 0"
        
    print("Top-P passed.")

def test_stability(device):
    print("Testing Numerical Stability (Low Temp)...")
    # Logits with large difference: 10 vs 20.
    # At temp=0.01, difference becomes (10-20)/0.01 = -1000. exp(-1000) is 0.
    # Implementation must handle this without overflow/inf issues.
    logits_np = np.array([[10.0, 20.0]], dtype=np.float32)
    logits_torch = torch.from_numpy(logits_np)
    
    logits_llaisys = create_llaisys_tensor(logits_torch, device)
    out_token, out_token_llaisys = zero_tensor((1,), "i64", device)
    workspace = create_workspace(device)

    try:
        llaisys.Ops.random_sample(out_token_llaisys, logits_llaisys, workspace, 0.01, 1.0, 0)
        token_id = get_output_token(out_token_llaisys, device)
        
        # Should deterministically pick the max value (index 1) at very low temp
        assert token_id == 1, f"Stability failed: got {token_id}, expected 1"
        print("Stability passed.")
    except Exception as e:
        print(f"Stability failed with exception: {e}")
        raise e

def test_determinism(device):
    print("Testing Determinism (expecting different results)...")
    shape = (1, 100)
    # Uniform distribution - any index is equally likely
    logits_torch = torch.zeros(shape) 
    
    logits_llaisys = create_llaisys_tensor(logits_torch, device)
    out_token, out_token_llaisys = zero_tensor((1,), "i64", device)
    workspace = create_workspace(device)

    results = []
    for _ in range(20):
        llaisys.Ops.random_sample(out_token_llaisys, logits_llaisys, workspace, 1.0, 1.0, 0)
        token_id = get_output_token(out_token_llaisys, device)
        results.append(token_id)
        
    unique_results = len(set(results))
    print(f"Unique results in 20 runs: {unique_results}")
    assert unique_results > 1, "Determinism failed: output is constant (randomness missing)"
    print("Determinism check passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()
    
    print(f"Running extended random_sample tests on {args.device}")
    
    try:
        test_top_k(args.device)
        test_top_p(args.device)
        test_stability(args.device)
        test_determinism(args.device)
        print("\033[92mAll extended tests passed!\033[0m")
        sys.exit(0)
    except Exception as e:
        print(f"\033[91mTests FAILED: {e}\033[0m")
        sys.exit(1)
