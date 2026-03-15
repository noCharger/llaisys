import sys
import os
import argparse
import ctypes

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
# Also add the python directory to path so llaisys can be imported if not installed
python_dir = os.path.abspath(os.path.join(parent_dir, "../python"))
sys.path.insert(0, python_dir)

import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark, zero_tensor


def torch_random_sample(out_token, logits, temp, top_p, top_k):
    # 1. Temperature
    if temp > 0:
        logits = logits / temp
        
    probs = torch.softmax(logits, dim=-1)
    
    # 2. Top-K
    if top_k > 0:
        top_k = min(top_k, probs.size(-1))
        vals, _ = torch.topk(probs, top_k)
        min_val = vals[:, -1].unsqueeze(-1)
        probs = torch.where(probs < min_val, torch.tensor(0.0, device=probs.device, dtype=probs.dtype), probs)
        
    # 3. Top-P
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs = probs.masked_fill(indices_to_remove, 0.0)
        
    # 4. Sample
    # Normalize
    probs_sum = torch.sum(probs, dim=-1, keepdim=True)
    probs = torch.where(probs_sum > 0, probs / probs_sum, probs)
    
    if torch.sum(probs) > 0:
        next_token = torch.multinomial(probs, num_samples=1)
        out_token.copy_(next_token.flatten())
    else:
        out_token.copy_(torch.argmax(logits, dim=-1))


def test_op_random_sample(
    shape,
    dtype_name="f32",
    device_name="cpu",
    profile=False,
    temp=0.7,
    top_p=0.9,
    top_k=40
):
    print(f"   shape {shape} dtype <{dtype_name}> temp={temp} top_p={top_p} top_k={top_k}")
    
    logits, logits_llaisys = random_tensor(shape, dtype_name, device_name)
    out_token, out_token_llaisys = zero_tensor((1,), "i64", device_name)
    out_token_torch = torch.zeros((1,), dtype=torch.int64, device=logits.device)

    torch_random_sample(out_token_torch, logits, temp, top_p, top_k)
    llaisys.Ops.random_sample(out_token_llaisys, logits_llaisys, temp, top_p, top_k)

    # Check validity (bounds check)
    data_ptr = out_token_llaisys.data_ptr()
    token_id = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_int64)).contents.value
    
    assert 0 <= token_id < shape[1], f"Token ID {token_id} out of bounds [0, {shape[1]})"

    if profile:
        compiled_torch_sample = torch.compile(torch_random_sample)
        benchmark(
            lambda: torch_random_sample(out_token_torch, logits, temp, top_p, top_k),
            lambda: llaisys.Ops.random_sample(out_token_llaisys, logits_llaisys, temp, top_p, top_k),
            device_name,
            torch_compile_func=lambda: compiled_torch_sample(out_token_torch, logits, temp, top_p, top_k)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    
    testShapes = [(1, 32000), (1, 128000)]
    testDtype = ["f32", "f16", "bf16"]
    
    print(f"Testing Ops.random_sample on {args.device}")
    for shape in testShapes:
        for dtype_name in testDtype:
            test_op_random_sample(shape, dtype_name, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
