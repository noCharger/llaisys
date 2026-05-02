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

    def forward_batch_paged(
        self,
        pool: "PagedKVCache",
        packed_tokens: list,
        cu_seqlens_q: list,
        block_ids: list,
        slot_mapping: list,
        temps: list,
        top_ps: list,
        top_ks: list,
    ) -> list:
        """Paged-backend forward. Caller must have already called pool.append
        for each block to grow its page_table; slot_mapping is the
        concatenation of those returns in the order of block_ids (each
        contributing q_len entries)."""
        batch = len(block_ids)
        assert len(cu_seqlens_q) == batch + 1
        assert len(temps) == len(top_ps) == len(top_ks) == batch
        total_q = cu_seqlens_q[-1]
        assert len(packed_tokens) == total_q
        assert len(slot_mapping) == total_q

        tok_arr = (ctypes.c_int64 * total_q)(*packed_tokens)
        cu_arr = (ctypes.c_int32 * (batch + 1))(*cu_seqlens_q)
        bid_arr = (ctypes.c_int32 * batch)(*block_ids)
        slot_arr = (ctypes.c_int32 * total_q)(*slot_mapping)
        temp_arr = (ctypes.c_float * batch)(*temps)
        top_p_arr = (ctypes.c_float * batch)(*top_ps)
        top_k_arr = (ctypes.c_int * batch)(*top_ks)
        out_arr = (ctypes.c_int64 * batch)()

        rc = LIB_LLAISYS.llaisysQwen2ModelForwardBatchPaged(
            self.model, pool._pool,
            tok_arr, cu_arr, bid_arr, slot_arr,
            temp_arr, top_p_arr, top_k_arr,
            ctypes.c_int32(batch), out_arr,
        )
        if rc != 0:
            raise RuntimeError(
                f"forward_batch_paged failed with code {rc} "
                "(see stderr for details)")
        return [int(x) for x in out_arr]

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


class PagedKVCache:
    """PagedAttention-style KV pool with per-tenant quota.

    The underlying C pool keeps a single shared physical-page array (one big
    K/V tensor per layer) and per-request block tables. Same-tenant prefix
    sharing is supported via a chain-hash index (16-token chunks). Cross-tenant
    pages are zero-wiped on takeover; CoW never crosses tenants.
    """

    def __init__(self, model: Qwen2, n_pages: int, page_size: int = 16,
                 max_pages_per_request: int = 1024):
        self._pool = LIB_LLAISYS.llaisysQwen2PagedPoolCreate(
            model.model,
            ctypes.c_size_t(n_pages),
            ctypes.c_size_t(page_size),
            ctypes.c_size_t(max_pages_per_request),
        )
        if not self._pool:
            raise RuntimeError("Failed to create PagedKVCache")

    def __del__(self):
        if hasattr(self, "_pool") and self._pool:
            LIB_LLAISYS.llaisysQwen2PagedPoolDestroy(self._pool)
            self._pool = None

    @staticmethod
    def _hash_tenant(tenant_id) -> int:
        if isinstance(tenant_id, int):
            v = tenant_id
        else:
            import hashlib
            h = hashlib.sha256(str(tenant_id).encode("utf-8")).digest()
            v = int.from_bytes(h[:8], "little", signed=False)
        if v == 0:
            v = 1
        return v & 0xFFFFFFFFFFFFFFFF

    def set_tenant_quota(self, tenant_id, reservation_floor: int,
                         max_pages: int, burst_pages: int):
        tid = self._hash_tenant(tenant_id)
        LIB_LLAISYS.llaisysQwen2PagedPoolSetTenantQuota(
            self._pool, ctypes.c_uint64(tid),
            ctypes.c_size_t(reservation_floor),
            ctypes.c_size_t(max_pages),
            ctypes.c_size_t(burst_pages),
        )

    def acquire(self, tenant_id, prefix_tokens=None):
        """Returns (block_id, matched_prefix_len). matched_prefix_len is in tokens
        and is always a multiple of page_size."""
        tid = self._hash_tenant(tenant_id)
        prefix_tokens = prefix_tokens or []
        n = len(prefix_tokens)
        arr = (ctypes.c_int64 * n)(*prefix_tokens) if n > 0 else None
        matched = ctypes.c_size_t(0)
        bid = LIB_LLAISYS.llaisysQwen2PagedPoolAcquire(
            self._pool, ctypes.c_uint64(tid),
            arr, ctypes.c_size_t(n),
            ctypes.byref(matched),
        )
        return int(bid), int(matched.value)

    def release(self, block_id: int):
        LIB_LLAISYS.llaisysQwen2PagedPoolRelease(self._pool, ctypes.c_int32(block_id))

    def append(self, block_id: int, n_new_tokens: int):
        """Allocate pages for `n_new_tokens` new tokens. Returns the list of
        packed slots (high16 = page_id, low16 = offset). Raises RuntimeError
        on quota / pool exhaustion."""
        if n_new_tokens == 0:
            return []
        slots = (ctypes.c_int32 * n_new_tokens)()
        rc = LIB_LLAISYS.llaisysQwen2PagedPoolAppend(
            self._pool, ctypes.c_int32(block_id),
            ctypes.c_size_t(n_new_tokens), slots,
        )
        if rc != 0:
            raise RuntimeError(f"PagedKVCache.append failed (rc={rc}); pool/quota exhausted")
        return [int(s) for s in slots]

    def commit(self, block_id: int, new_pos: int, tokens):
        n = len(tokens)
        arr = (ctypes.c_int64 * n)(*tokens) if n > 0 else None
        LIB_LLAISYS.llaisysQwen2PagedPoolCommit(
            self._pool, ctypes.c_int32(block_id),
            ctypes.c_size_t(new_pos), arr, ctypes.c_size_t(n),
        )

    def block_pos(self, block_id: int) -> int:
        return int(LIB_LLAISYS.llaisysQwen2PagedPoolBlockPos(
            self._pool, ctypes.c_int32(block_id)))

    def page_table(self, block_id: int):
        # First call to get size; second to fill.
        size = LIB_LLAISYS.llaisysQwen2PagedPoolPageTable(
            self._pool, ctypes.c_int32(block_id), None)
        n = int(size)
        if n == 0:
            return []
        buf = (ctypes.c_int32 * n)()
        LIB_LLAISYS.llaisysQwen2PagedPoolPageTable(
            self._pool, ctypes.c_int32(block_id), buf)
        return [int(x) for x in buf]

    def page_k(self, page_id: int, layer: int):
        return LIB_LLAISYS.llaisysQwen2PagedPoolPageK(
            self._pool, ctypes.c_int32(page_id), ctypes.c_size_t(layer))

    def page_v(self, page_id: int, layer: int):
        return LIB_LLAISYS.llaisysQwen2PagedPoolPageV(
            self._pool, ctypes.c_int32(page_id), ctypes.c_size_t(layer))

    @property
    def n_pages(self) -> int:
        return int(LIB_LLAISYS.llaisysQwen2PagedPoolNumPages(self._pool))

    @property
    def page_size(self) -> int:
        return int(LIB_LLAISYS.llaisysQwen2PagedPoolPageSize(self._pool))

    def tenant_pages_used(self, tenant_id) -> int:
        tid = self._hash_tenant(tenant_id)
        return int(LIB_LLAISYS.llaisysQwen2PagedPoolTenantPagesUsed(
            self._pool, ctypes.c_uint64(tid)))

    @property
    def global_pages_free(self) -> int:
        return int(LIB_LLAISYS.llaisysQwen2PagedPoolGlobalPagesFree(self._pool))

    @staticmethod
    def unpack_slot(slot: int):
        """Unpack a slot integer to (page_id, offset)."""
        return (slot >> 16) & 0xFFFF, slot & 0xFFFF

    def scatter_kv(self, layer: int, k_new, v_new, slot_mapping):
        """Write packed K_new[n_tokens, nkvh, dh] / V_new into pool pages at
        the given slots. Used by the model forward path each layer."""
        n = len(slot_mapping)
        slot_arr = (ctypes.c_int32 * n)(*slot_mapping)
        rc = LIB_LLAISYS.llaisysQwen2PagedPoolScatterKV(
            self._pool, ctypes.c_size_t(layer),
            k_new.lib_tensor() if hasattr(k_new, "lib_tensor") else k_new,
            v_new.lib_tensor() if hasattr(v_new, "lib_tensor") else v_new,
            slot_arr, ctypes.c_size_t(n))
        if rc != 0:
            raise RuntimeError(f"PagedKVCache.scatter_kv failed (rc={rc})")
