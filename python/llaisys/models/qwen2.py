from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights

from pathlib import Path
import safetensors.numpy
import safetensors
import json
import ctypes
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU, dtype: DataType = DataType.F32):
        model_path = Path(model_path)
        
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
            
        self.meta = LlaisysQwen2Meta()
        self.meta.dtype = dtype
        
        self.meta.nlayer = config["num_hidden_layers"]
        self.meta.hs = config["hidden_size"]
        self.meta.nh = config["num_attention_heads"]
        self.meta.nkvh = config["num_key_value_heads"]
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = config["intermediate_size"]
        self.meta.maxseq = config.get("max_position_embeddings", 8192)
        self.meta.voc = config["vocab_size"]
        self.meta.epsilon = config["rms_norm_eps"]
        self.meta.theta = config.get("rope_theta", 10000.0)
        self.meta.end_token = config.get("eos_token_id", 151643)
        
        self.end_token = self.meta.end_token

        # Create Model
        device_ids = (ctypes.c_int * 1)(0)
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(ctypes.byref(self.meta), device, device_ids, 1)
        
        if not self.model:
            raise RuntimeError("Failed to create model")
            
        self.weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model).contents

        for file in sorted(model_path.glob("*.safetensors")):
            framework = "pt" if HAS_TORCH else "numpy"
            with safetensors.safe_open(file, framework=framework, device="cpu") as f:
                for name in f.keys():
                    ptr = self._get_tensor_ptr(name)
                    if ptr:
                        data = f.get_tensor(name)
                        
                        # Handle Torch Tensors (converts bf16 if needed)
                        if HAS_TORCH and isinstance(data, torch.Tensor):
                            if self.meta.dtype == DataType.F16:
                                data = data.half().numpy()
                            else:
                                data = data.float().numpy()
                            
                        # Handle Numpy (ensure correct dtype)
                        if isinstance(data, np.ndarray):
                            target_dtype = np.float32
                            if self.meta.dtype == DataType.F16:
                                target_dtype = np.float16
                            
                            if data.dtype != target_dtype:
                                data = data.astype(target_dtype)
                            if not data.flags['C_CONTIGUOUS']:
                                data = np.ascontiguousarray(data)
                        
                        LIB_LLAISYS.tensorLoad(ptr, data.ctypes.data_as(ctypes.c_void_p))

    def _get_tensor_ptr(self, name):
        if name == "model.embed_tokens.weight":
            return self.weights.in_embed
        if name == "lm_head.weight":
            return self.weights.out_embed
        if name == "model.norm.weight":
            return self.weights.out_norm_w
        
        if name.startswith("model.layers."):
            parts = name.split(".")
            try:
                layer_idx = int(parts[2])
            except ValueError:
                return None
            
            subname = ".".join(parts[3:])
            
            if subname == "input_layernorm.weight":
                return self.weights.attn_norm_w[layer_idx]
            if subname == "post_attention_layernorm.weight":
                return self.weights.mlp_norm_w[layer_idx]
            
            if subname == "self_attn.q_proj.weight":
                return self.weights.attn_q_w[layer_idx]
            if subname == "self_attn.q_proj.bias":
                return self.weights.attn_q_b[layer_idx]
            if subname == "self_attn.k_proj.weight":
                return self.weights.attn_k_w[layer_idx]
            if subname == "self_attn.k_proj.bias":
                return self.weights.attn_k_b[layer_idx]
            if subname == "self_attn.v_proj.weight":
                return self.weights.attn_v_w[layer_idx]
            if subname == "self_attn.v_proj.bias":
                return self.weights.attn_v_b[layer_idx]
            if subname == "self_attn.o_proj.weight":
                return self.weights.attn_o_w[layer_idx]
            
            if subname == "mlp.gate_proj.weight":
                return self.weights.mlp_gate_w[layer_idx]
            if subname == "mlp.up_proj.weight":
                return self.weights.mlp_up_w[layer_idx]
            if subname == "mlp.down_proj.weight":
                return self.weights.mlp_down_w[layer_idx]
        
        return None

    def create_session(self):
        return LIB_LLAISYS.llaisysQwen2ModelCreateSession(self.model)

    def destroy_session(self, session_ptr):
        LIB_LLAISYS.llaisysQwen2ModelDestroySession(session_ptr)

    def rewind_session(self, session_ptr, length):
        LIB_LLAISYS.llaisysQwen2ModelRewindSession(session_ptr, length)

    def forward(self, session_ptr, input_ids, temperature=0.7, top_p=0.9, top_k=40):
        if not input_ids:
             return self.end_token
        
        arr = (ctypes.c_int64 * len(input_ids))(*input_ids)
        return LIB_LLAISYS.llaisysQwen2ModelForward(
            session_ptr, 
            arr, 
            len(input_ids), 
            temperature, 
            top_p, 
            top_k
        )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if not inputs:
            return []
            
        tokens = list(inputs)
        session = self.create_session()
        
        try:
            # Prefill
            input_array = (ctypes.c_int64 * len(tokens))(*tokens)
            next_token = LIB_LLAISYS.llaisysQwen2ModelForward(session, input_array, len(tokens), temperature, top_p, top_k)
            
            tokens.append(next_token)
            if next_token == self.end_token:
                return tokens
            
            for _ in range(max_new_tokens - 1):
                input_array = (ctypes.c_int64 * 1)(tokens[-1])
                next_token = LIB_LLAISYS.llaisysQwen2ModelForward(session, input_array, 1, temperature, top_p, top_k)
                tokens.append(next_token)
                
                if next_token == self.end_token:
                    break
        finally:
            self.destroy_session(session)
                
        return tokens
        
    def __del__(self):
        if hasattr(self, 'model') and self.model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
