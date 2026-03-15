import gc
from test_utils import *

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import time
import llaisys
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def load_hf_model(model_path=None, device_name="cpu"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )

    return tokenizer, model, model_path


def hf_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs[0].tolist(), result


def load_llaisys_model(model_path, device_name):
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    return model


def llaisys_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    return outputs, tokenizer.decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_sampling", action="store_true", help="Test random sampling behaviors")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--repeat", default=10, type=int, help="Number of repeats for benchmark")
    parser.add_argument("--profile", action="store_true", help="Show e2e inference time profile")

    args = parser.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    tokenizer, model, model_path = load_hf_model(args.model, args.device)

    # Example prompt
    if args.device == "nvidia":
        torch.cuda.synchronize()
    start_time = time.time()
    tokens, output = hf_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    if args.device == "nvidia":
        torch.cuda.synchronize()
    end_time = time.time()
    hf_duration = end_time - start_time

    del model
    gc.collect()

    print("\n=== Answer ===\n")
    print("Tokens:")
    print(tokens)
    print("\nContents:")
    print(output)
    print("\n")
    print(f"Time elapsed: {hf_duration:.2f}s\n")

    model = load_llaisys_model(model_path, args.device)
    if args.device == "nvidia":
        torch.cuda.synchronize()
    start_time = time.time()
    llaisys_tokens, llaisys_output = llaisys_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    if args.device == "nvidia":
        torch.cuda.synchronize()
    end_time = time.time()
    llaisys_duration = end_time - start_time

    print("\n=== Your Result ===\n")
    print("Tokens:")
    print(llaisys_tokens)
    print("\nContents:")
    print(llaisys_output)
    print("\n")
    print(f"Time elapsed: {llaisys_duration:.2f}s\n")

    if args.profile:
        print("\n=== Profile ===\n")
        print(f"HF E2E Inference Time: {hf_duration:.4f}s")
        print(f"Llaisys E2E Inference Time: {llaisys_duration:.4f}s")

    if args.benchmark:
        print("\n=== Benchmarking ===\n")
        
        # Warmup
        print("Warming up...")
        for _ in range(3):
            llaisys_infer(args.prompt, tokenizer, model, 10, top_p, top_k, temperature)
            
        latencies = []
        tokens_per_sec = []
        
        for i in range(args.repeat):
            start = time.time()
            out_tokens, _ = llaisys_infer(args.prompt, tokenizer, model, args.max_steps, top_p, top_k, temperature)
            if args.device == "nvidia":
                torch.cuda.synchronize()
            end = time.time()
            
            latency = (end - start) * 1000
            tps = len(out_tokens) / (end - start)
            
            latencies.append(latency)
            tokens_per_sec.append(tps)
            print(f"Iter {i+1}: {latency:.2f}ms, {tps:.2f} tokens/s")
            
        import numpy as np
        print(f"\nDevice: {args.device}")
        print(f"End-to-End Latency:")
        print(f"  Mean: {np.mean(latencies):.2f} ms")
        print(f"  P50:  {np.percentile(latencies, 50):.2f} ms")
        print(f"  P90:  {np.percentile(latencies, 90):.2f} ms")
        print(f"  P99:  {np.percentile(latencies, 99):.2f} ms")
        print(f"Throughput:")
        print(f"  Mean: {np.mean(tokens_per_sec):.2f} tokens/s")
        
    if args.test:
        assert llaisys_tokens == tokens
        print("\033[92mTest passed!\033[0m\n")

    if args.test_sampling:
        print("\n=== Testing Sampling Behaviors ===\n")
        
        # 1. Verify output same with deterministic params (top_k=1)
        print("1. Verifying deterministic generation (top_k=1)...")
        torch.manual_seed(42)
        det_tokens_1, det_out_1 = llaisys_infer(
            args.prompt, tokenizer, model, args.max_steps, top_p=1.0, top_k=1, temperature=1.0
        )
        torch.manual_seed(42)
        det_tokens_2, det_out_2 = llaisys_infer(
            args.prompt, tokenizer, model, args.max_steps, top_p=1.0, top_k=1, temperature=1.0
        )
        assert det_tokens_1 == det_tokens_2, "Outputs should be exactly the same for top_k=1"
        print("   Deterministic check passed.\n")

        # 2. Change parameters, output should be diff
        print("2. Verifying random generation (top_k=50, top_p=0.9, temp=1.5)...")
        rand_tokens_1, rand_out_1 = llaisys_infer(
            args.prompt, tokenizer, model, args.max_steps, top_p=0.9, top_k=50, temperature=1.5
        )
        rand_tokens_2, rand_out_2 = llaisys_infer(
            args.prompt, tokenizer, model, args.max_steps, top_p=0.9, top_k=50, temperature=1.5
        )
        if det_tokens_1 == rand_tokens_1:
            print("Warning: Sampling output was identical to greedy output. This can happen by chance, but retrying with higher temp...")
            rand_tokens_1, rand_out_1 = llaisys_infer(
                args.prompt, tokenizer, model, args.max_steps, top_p=0.9, top_k=50, temperature=2.5
            )
            
        assert det_tokens_1 != rand_tokens_1, "Sampling output should differ from greedy output"
        print("   Random sampling check passed.\n")
        
        print("\033[92mSampling test passed!\033[0m\n")

