import ctypes
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t

class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]

class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_o_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_gate_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_up_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_down_w", ctypes.POINTER(llaisysTensor_t)),
    ]

def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [ctypes.POINTER(LlaisysQwen2Meta), llaisysDeviceType_t, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.llaisysQwen2ModelCreate.restype = ctypes.c_void_p

    lib.llaisysQwen2ModelDestroy.argtypes = [ctypes.c_void_p]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [ctypes.c_void_p]
    lib.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelInfer.argtypes = [
        ctypes.c_void_p, 
        ctypes.POINTER(ctypes.c_int64), 
        ctypes.c_size_t,
        ctypes.c_float, 
        ctypes.c_float, 
        ctypes.c_int
    ]
    lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64

    # Session APIs
    lib.llaisysQwen2ModelCreateSession.argtypes = [ctypes.c_void_p]
    lib.llaisysQwen2ModelCreateSession.restype = ctypes.c_void_p

    lib.llaisysQwen2ModelDestroySession.argtypes = [ctypes.c_void_p]
    lib.llaisysQwen2ModelDestroySession.restype = None

    lib.llaisysQwen2ModelRewindSession.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.llaisysQwen2ModelRewindSession.restype = None

    lib.llaisysQwen2ModelForward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int
    ]
    lib.llaisysQwen2ModelForward.restype = ctypes.c_int64

    # ----- Paged KV Pool -----
    lib.llaisysQwen2PagedPoolCreate.argtypes = [
        ctypes.c_void_p,    # model
        ctypes.c_size_t,    # n_pages
        ctypes.c_size_t,    # page_size
        ctypes.c_size_t,    # max_pages_per_request
    ]
    lib.llaisysQwen2PagedPoolCreate.restype = ctypes.c_void_p

    lib.llaisysQwen2PagedPoolDestroy.argtypes = [ctypes.c_void_p]
    lib.llaisysQwen2PagedPoolDestroy.restype = None

    lib.llaisysQwen2PagedPoolSetTenantQuota.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]
    lib.llaisysQwen2PagedPoolSetTenantQuota.restype = None

    lib.llaisysQwen2PagedPoolAcquire.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.llaisysQwen2PagedPoolAcquire.restype = ctypes.c_int32

    lib.llaisysQwen2PagedPoolRelease.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.llaisysQwen2PagedPoolRelease.restype = None

    lib.llaisysQwen2PagedPoolAppend.argtypes = [
        ctypes.c_void_p, ctypes.c_int32,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.llaisysQwen2PagedPoolAppend.restype = ctypes.c_int32

    lib.llaisysQwen2PagedPoolCommit.argtypes = [
        ctypes.c_void_p, ctypes.c_int32,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t,
    ]
    lib.llaisysQwen2PagedPoolCommit.restype = None

    lib.llaisysQwen2PagedPoolBlockPos.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.llaisysQwen2PagedPoolBlockPos.restype = ctypes.c_size_t

    lib.llaisysQwen2PagedPoolPageTable.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
    ]
    lib.llaisysQwen2PagedPoolPageTable.restype = ctypes.c_size_t

    lib.llaisysQwen2PagedPoolPageK.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_size_t]
    lib.llaisysQwen2PagedPoolPageK.restype = llaisysTensor_t
    lib.llaisysQwen2PagedPoolPageV.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_size_t]
    lib.llaisysQwen2PagedPoolPageV.restype = llaisysTensor_t

    lib.llaisysQwen2PagedPoolNumPages.argtypes = [ctypes.c_void_p]
    lib.llaisysQwen2PagedPoolNumPages.restype = ctypes.c_size_t
    lib.llaisysQwen2PagedPoolPageSize.argtypes = [ctypes.c_void_p]
    lib.llaisysQwen2PagedPoolPageSize.restype = ctypes.c_size_t
    lib.llaisysQwen2PagedPoolTenantPagesUsed.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.llaisysQwen2PagedPoolTenantPagesUsed.restype = ctypes.c_size_t
    lib.llaisysQwen2PagedPoolGlobalPagesFree.argtypes = [ctypes.c_void_p]
    lib.llaisysQwen2PagedPoolGlobalPagesFree.restype = ctypes.c_size_t

    lib.llaisysQwen2PagedPoolScatterKV.argtypes = [
        ctypes.c_void_p,                 # pool
        ctypes.c_size_t,                 # layer
        llaisysTensor_t,                 # k_new [n_tokens, nkvh, dh]
        llaisysTensor_t,                 # v_new
        ctypes.POINTER(ctypes.c_int32),  # slot_mapping
        ctypes.c_size_t,                 # n_tokens
    ]
    lib.llaisysQwen2PagedPoolScatterKV.restype = ctypes.c_int32

    lib.llaisysQwen2PagedPoolBigK.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.llaisysQwen2PagedPoolBigK.restype = llaisysTensor_t
    lib.llaisysQwen2PagedPoolBigV.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.llaisysQwen2PagedPoolBigV.restype = llaisysTensor_t

    lib.llaisysQwen2ModelForwardBatchPaged.argtypes = [
        ctypes.c_void_p,                # model
        ctypes.c_void_p,                # paged pool
        ctypes.POINTER(ctypes.c_int64), # packed_tokens
        ctypes.POINTER(ctypes.c_int32), # cu_seqlens_q
        ctypes.POINTER(ctypes.c_int32), # block_ids
        ctypes.POINTER(ctypes.c_int32), # slot_mapping
        ctypes.POINTER(ctypes.c_float), # temps
        ctypes.POINTER(ctypes.c_float), # top_ps
        ctypes.POINTER(ctypes.c_int),   # top_ks
        ctypes.c_int32,                 # batch
        ctypes.POINTER(ctypes.c_int64), # out_next_tokens
    ]
    lib.llaisysQwen2ModelForwardBatchPaged.restype = ctypes.c_int32
