#include "llaisys/models/qwen2.h"
#include "llaisys/ops.h"
#include "llaisys/tensor.h"
#include "llaisys/runtime.h"
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    llaisysDeviceType_t device_type;
    int device_id;

    // KV Cache: nlayer of tensors
    // Key-Value Cache stores past token representations for each layer
    // Each tensor shape: [max_seq, nkvh, dh]
    std::vector<llaisysTensor_t> k_cache;
    std::vector<llaisysTensor_t> v_cache;
    
    // Current filled position in KV cache
    size_t pos;

    llaisysTensor_t zero_bias_hs;
    llaisysTensor_t zero_bias_di;
    llaisysTensor_t zero_bias_voc;
};

static llaisysTensor_t create_tensor(size_t* shape, size_t ndim, llaisysDataType_t dtype, llaisysDeviceType_t device, int device_id) {
    llaisysTensor_t t = tensorCreate(shape, ndim, dtype, device, device_id);
    
    size_t elem_size = (dtype == LLAISYS_DTYPE_F32) ? 4 : 2;
    if (dtype == LLAISYS_DTYPE_BF16) elem_size = 2;
    if (dtype == LLAISYS_DTYPE_I64) elem_size = 8;
    
    size_t numel = 1;
    for (size_t i = 0; i < ndim; ++i) numel *= shape[i];
    
    size_t byte_size = numel * elem_size;
    
    if (device == LLAISYS_DEVICE_CPU) {
        void* ptr = tensorGetData(t);
        memset(ptr, 0, byte_size);
    } else {
        void* host_zeros = calloc(1, byte_size);
        const auto runtime = llaisysGetRuntimeAPI(device);
        runtime->memcpy_sync(tensorGetData(t), host_zeros, byte_size, LLAISYS_MEMCPY_H2D);
        free(host_zeros);
    }
    
    return t;
}

// Helper to create weight tensor with specific shape
static llaisysTensor_t create_weight(const LlaisysQwen2Meta* meta, const size_t* shape, size_t ndim, llaisysDeviceType_t device, int device_id) {
    return create_tensor((size_t*)shape, ndim, meta->dtype, device, device_id);
}

// Helper to create and initialize zero tensor
static llaisysTensor_t create_zero_tensor(size_t size, llaisysDataType_t dtype, llaisysDeviceType_t device, int device_id) {
    size_t shape[1] = {size};
    llaisysTensor_t t = tensorCreate(shape, 1, dtype, device, device_id);
    
    size_t elem_size = (dtype == LLAISYS_DTYPE_F32) ? 4 : 2;
    if (dtype == LLAISYS_DTYPE_BF16) elem_size = 2;
    
    size_t byte_size = size * elem_size;
    void* host_zeros = calloc(1, byte_size);
    
    const auto runtime = llaisysGetRuntimeAPI(device);
    runtime->memcpy_sync(tensorGetData(t), host_zeros, byte_size, LLAISYS_MEMCPY_H2D);
    
    free(host_zeros);
    return t;
}

extern "C" {

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    LlaisysQwen2Model* model = new LlaisysQwen2Model();
    model->meta = *meta;
    model->device_type = device;
    model->device_id = (ndevice > 0) ? device_ids[0] : 0;
    model->pos = 0;

    model->zero_bias_hs = create_zero_tensor(meta->hs, meta->dtype, device, model->device_id);
    model->zero_bias_di = create_zero_tensor(meta->di, meta->dtype, device, model->device_id);
    model->zero_bias_voc = create_zero_tensor(meta->voc, meta->dtype, device, model->device_id);

    // Allocate weights structure arrays
    model->weights.attn_norm_w = (llaisysTensor_t*)malloc(sizeof(llaisysTensor_t) * meta->nlayer);
    model->weights.attn_q_w = (llaisysTensor_t*)malloc(sizeof(llaisysTensor_t) * meta->nlayer);
    model->weights.attn_q_b = (llaisysTensor_t*)malloc(sizeof(llaisysTensor_t) * meta->nlayer);
    model->weights.attn_k_w = (llaisysTensor_t*)malloc(sizeof(llaisysTensor_t) * meta->nlayer);
    model->weights.attn_k_b = (llaisysTensor_t*)malloc(sizeof(llaisysTensor_t) * meta->nlayer);
    model->weights.attn_v_w = (llaisysTensor_t*)malloc(sizeof(llaisysTensor_t) * meta->nlayer);
    model->weights.attn_v_b = (llaisysTensor_t*)malloc(sizeof(llaisysTensor_t) * meta->nlayer);
    model->weights.attn_o_w = (llaisysTensor_t*)malloc(sizeof(llaisysTensor_t) * meta->nlayer);
    model->weights.mlp_norm_w = (llaisysTensor_t*)malloc(sizeof(llaisysTensor_t) * meta->nlayer);
    model->weights.mlp_gate_w = (llaisysTensor_t*)malloc(sizeof(llaisysTensor_t) * meta->nlayer);
    model->weights.mlp_up_w = (llaisysTensor_t*)malloc(sizeof(llaisysTensor_t) * meta->nlayer);
    model->weights.mlp_down_w = (llaisysTensor_t*)malloc(sizeof(llaisysTensor_t) * meta->nlayer);

    // Create tensors for weights
    size_t shape[3];

    // Embeddings
    shape[0] = meta->voc; shape[1] = meta->hs;
    model->weights.in_embed = create_weight(meta, shape, 2, device, model->device_id);
    model->weights.out_embed = create_weight(meta, shape, 2, device, model->device_id);

    // Norms
    shape[0] = meta->hs;
    model->weights.out_norm_w = create_weight(meta, shape, 1, device, model->device_id);

    // Layers
    for (size_t i = 0; i < meta->nlayer; ++i) {
        // Norms
        shape[0] = meta->hs;
        model->weights.attn_norm_w[i] = create_weight(meta, shape, 1, device, model->device_id);
        model->weights.mlp_norm_w[i] = create_weight(meta, shape, 1, device, model->device_id);

        // QKV
        // [nh * dh, hs] -> [out, in]
        size_t hidden = meta->hs;
        size_t q_dim = meta->nh * meta->dh;
        size_t kv_dim = meta->nkvh * meta->dh;
        size_t inter_dim = meta->di;

        // Q
        shape[0] = q_dim; shape[1] = hidden;
        model->weights.attn_q_w[i] = create_weight(meta, shape, 2, device, model->device_id);
        shape[0] = q_dim;
        model->weights.attn_q_b[i] = create_weight(meta, shape, 1, device, model->device_id);

        // K
        shape[0] = kv_dim; shape[1] = hidden;
        model->weights.attn_k_w[i] = create_weight(meta, shape, 2, device, model->device_id);
        shape[0] = kv_dim;
        model->weights.attn_k_b[i] = create_weight(meta, shape, 1, device, model->device_id);

        // V
        shape[0] = kv_dim; shape[1] = hidden;
        model->weights.attn_v_w[i] = create_weight(meta, shape, 2, device, model->device_id);
        shape[0] = kv_dim;
        model->weights.attn_v_b[i] = create_weight(meta, shape, 1, device, model->device_id);

        // O
        shape[0] = hidden; shape[1] = q_dim;
        model->weights.attn_o_w[i] = create_weight(meta, shape, 2, device, model->device_id);
        
        // MLP
        // Gate
        shape[0] = inter_dim; shape[1] = hidden;
        model->weights.mlp_gate_w[i] = create_weight(meta, shape, 2, device, model->device_id);
        // Up
        shape[0] = inter_dim; shape[1] = hidden;
        model->weights.mlp_up_w[i] = create_weight(meta, shape, 2, device, model->device_id);
        // Down
        shape[0] = hidden; shape[1] = inter_dim;
        model->weights.mlp_down_w[i] = create_weight(meta, shape, 2, device, model->device_id);

        // KV Cache
        // [max_seq, nkvh, dh]
        shape[0] = meta->maxseq; shape[1] = meta->nkvh; shape[2] = meta->dh;
        model->k_cache.push_back(create_weight(meta, shape, 3, device, model->device_id));
        model->v_cache.push_back(create_weight(meta, shape, 3, device, model->device_id));
    }

    return model;
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    if (!model) return;
    
    tensorDestroy(model->weights.in_embed);
    tensorDestroy(model->weights.out_embed);
    tensorDestroy(model->weights.out_norm_w);

    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        tensorDestroy(model->weights.attn_norm_w[i]);
        tensorDestroy(model->weights.attn_q_w[i]);
        tensorDestroy(model->weights.attn_q_b[i]);
        tensorDestroy(model->weights.attn_k_w[i]);
        tensorDestroy(model->weights.attn_k_b[i]);
        tensorDestroy(model->weights.attn_v_w[i]);
        tensorDestroy(model->weights.attn_v_b[i]);
        tensorDestroy(model->weights.attn_o_w[i]);
        tensorDestroy(model->weights.mlp_norm_w[i]);
        tensorDestroy(model->weights.mlp_gate_w[i]);
        tensorDestroy(model->weights.mlp_up_w[i]);
        tensorDestroy(model->weights.mlp_down_w[i]);
        
        tensorDestroy(model->k_cache[i]);
        tensorDestroy(model->v_cache[i]);
    }

    free(model->weights.attn_norm_w);
    free(model->weights.attn_q_w);
    free(model->weights.attn_q_b);
    free(model->weights.attn_k_w);
    free(model->weights.attn_k_b);
    free(model->weights.attn_v_w);
    free(model->weights.attn_v_b);
    free(model->weights.attn_o_w);
    free(model->weights.mlp_norm_w);
    free(model->weights.mlp_gate_w);
    free(model->weights.mlp_up_w);
    free(model->weights.mlp_down_w);

    tensorDestroy(model->zero_bias_hs);
    tensorDestroy(model->zero_bias_di);
    tensorDestroy(model->zero_bias_voc);

    delete model;
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
    return &model->weights;
}

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    // 1. Embedding
    // Input: [ntoken]
    // Output: [ntoken, hs]
    
    size_t seq = ntoken;
    size_t hs = model->meta.hs;

    size_t shape_in[2] = {seq, 1};
    llaisysTensor_t tokens = tensorCreate(shape_in, 1, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
    tensorLoad(tokens, token_ids);

    size_t shape_embed[3] = {seq, hs};
    llaisysTensor_t x = tensorCreate(shape_embed, 2, model->meta.dtype, model->device_type, model->device_id);
    
    llaisysEmbedding(x, tokens, model->weights.in_embed);
    tensorDestroy(tokens);
    
    // 2. Layers
    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        llaisysTensor_t x_norm = tensorCreate(shape_embed, 2, model->meta.dtype, model->device_type, model->device_id);
        llaisysRmsNorm(x_norm, x, model->weights.attn_norm_w[i], model->meta.epsilon);

        // QKV
        // q = x_norm @ q_w
        size_t shape_q[3] = {seq, model->meta.nh, model->meta.dh};
        size_t shape_kv[3] = {seq, model->meta.nkvh, model->meta.dh};
        
        // Linear outputs flattened [seq, dim]
        size_t shape_q_flat[2] = {seq, model->meta.nh * model->meta.dh};
        size_t shape_kv_flat[2] = {seq, model->meta.nkvh * model->meta.dh};

        llaisysTensor_t q = tensorCreate(shape_q_flat, 2, model->meta.dtype, model->device_type, model->device_id);
        llaisysTensor_t k = tensorCreate(shape_kv_flat, 2, model->meta.dtype, model->device_type, model->device_id);
        llaisysTensor_t v = tensorCreate(shape_kv_flat, 2, model->meta.dtype, model->device_type, model->device_id);

        llaisysLinear(q, x_norm, model->weights.attn_q_w[i], model->weights.attn_q_b[i]);
        llaisysLinear(k, x_norm, model->weights.attn_k_w[i], model->weights.attn_k_b[i]);
        llaisysLinear(v, x_norm, model->weights.attn_v_w[i], model->weights.attn_v_b[i]);

        // Reshape to [seq, nh, dh]
        llaisysTensor_t q_view = tensorView(q, shape_q, 3);
        llaisysTensor_t k_view = tensorView(k, shape_kv, 3);
        llaisysTensor_t v_view = tensorView(v, shape_kv, 3);

        // RoPE
        size_t shape_pos[1] = {seq};
        llaisysTensor_t pos_ids = tensorCreate(shape_pos, 1, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
        std::vector<int64_t> pos_data(seq);
        for (size_t p = 0; p < seq; ++p) {
            pos_data[p] = model->pos + p;
        }
        tensorLoad(pos_ids, pos_data.data());

        llaisysROPE(q_view, q_view, pos_ids, model->meta.theta);
        llaisysROPE(k_view, k_view, pos_ids, model->meta.theta);
        
        tensorDestroy(pos_ids);

        // Update KV Cache
        // k_view is [seq, nkvh, dh]
        // k_cache is [max_seq, nkvh, dh]        
        llaisysTensor_t k_cache_slot = tensorSlice(model->k_cache[i], 0, model->pos, model->pos + seq);
        llaisysTensor_t v_cache_slot = tensorSlice(model->v_cache[i], 0, model->pos, model->pos + seq);
        
        // Copy data
        void* src_k = tensorGetData(k_view);
        void* dst_k = tensorGetData(k_cache_slot);
        void* src_v = tensorGetData(v_view);
        void* dst_v = tensorGetData(v_cache_slot);
        
        // Calculate size
        // seq * nkvh * dh * sizeof(dtype)
        size_t elem_size = (model->meta.dtype == LLAISYS_DTYPE_F32) ? 4 : 2;
        if (model->meta.dtype == LLAISYS_DTYPE_BF16) elem_size = 2;
        
        size_t copy_size = seq * model->meta.nkvh * model->meta.dh * elem_size;

        const auto runtime = llaisysGetRuntimeAPI(model->device_type);
        runtime->memcpy_sync(dst_k, src_k, copy_size, LLAISYS_MEMCPY_D2D);
        runtime->memcpy_sync(dst_v, src_v, copy_size, LLAISYS_MEMCPY_D2D);

        tensorDestroy(k_cache_slot);
        tensorDestroy(v_cache_slot);

        // Attention
        llaisysTensor_t k_full = tensorSlice(model->k_cache[i], 0, 0, model->pos + seq);
        llaisysTensor_t v_full = tensorSlice(model->v_cache[i], 0, 0, model->pos + seq);
        
        // Output tensor [seq, nh, dh]
        llaisysTensor_t attn_out = tensorCreate(shape_q, 3, model->meta.dtype, model->device_type, model->device_id);
        
        float scale = 1.0f / sqrtf((float)model->meta.dh);
        llaisysSelfAttention(attn_out, q_view, k_full, v_full, scale);

        tensorDestroy(k_full);
        tensorDestroy(v_full);
        tensorDestroy(q_view);
        tensorDestroy(k_view);
        tensorDestroy(v_view);
        tensorDestroy(q);
        tensorDestroy(k);
        tensorDestroy(v);

        // Flatten attn_out [seq, nh*dh]
        llaisysTensor_t attn_out_flat = tensorView(attn_out, shape_q_flat, 2);
        
        // Output projection
        llaisysTensor_t h = tensorCreate(shape_embed, 2, model->meta.dtype, model->device_type, model->device_id);
        
        llaisysLinear(h, attn_out_flat, model->weights.attn_o_w[i], model->zero_bias_hs);
        
        tensorDestroy(attn_out);
        tensorDestroy(attn_out_flat);

        // Residual Add: x = x + h
        llaisysAdd(x, x, h);
        tensorDestroy(h);
        tensorDestroy(x_norm); // x was used in residual, x_norm was temp.

        // MLP
        llaisysTensor_t mlp_norm = tensorCreate(shape_embed, 2, model->meta.dtype, model->device_type, model->device_id);
        llaisysRmsNorm(mlp_norm, x, model->weights.mlp_norm_w[i], model->meta.epsilon);

        // Gate & Up
        // Gate: [seq, di]
        size_t shape_inter[2] = {seq, model->meta.di};
        llaisysTensor_t gate = tensorCreate(shape_inter, 2, model->meta.dtype, model->device_type, model->device_id);
        llaisysTensor_t up = tensorCreate(shape_inter, 2, model->meta.dtype, model->device_type, model->device_id);
        
        llaisysLinear(gate, mlp_norm, model->weights.mlp_gate_w[i], model->zero_bias_di);
        llaisysLinear(up, mlp_norm, model->weights.mlp_up_w[i], model->zero_bias_di);

        // SwiGLU: out = swish(gate) * up
        llaisysTensor_t mlp_act = tensorCreate(shape_inter, 2, model->meta.dtype, model->device_type, model->device_id);
        llaisysSwiGLU(mlp_act, gate, up);

        tensorDestroy(gate);
        tensorDestroy(up);

        // Down
        llaisysTensor_t mlp_out = tensorCreate(shape_embed, 2, model->meta.dtype, model->device_type, model->device_id);
        llaisysLinear(mlp_out, mlp_act, model->weights.mlp_down_w[i], model->zero_bias_hs);

        tensorDestroy(mlp_act);
        
        // Residual
        llaisysAdd(x, x, mlp_out);
        
        tensorDestroy(mlp_out);
        tensorDestroy(mlp_norm);
    }

    // Final Norm
    llaisysTensor_t x_final = tensorCreate(shape_embed, 2, model->meta.dtype, model->device_type, model->device_id);
    llaisysRmsNorm(x_final, x, model->weights.out_norm_w, model->meta.epsilon);
    tensorDestroy(x); // Free previous x

    // Slice last row of x_final: [1, hs]
    llaisysTensor_t last_token_state = tensorSlice(x_final, 0, seq - 1, seq);
    
    // out_embed: [voc, hs]
    // logits: [1, voc]
    size_t shape_logits[2] = {1, model->meta.voc};
    llaisysTensor_t logits = tensorCreate(shape_logits, 2, model->meta.dtype, model->device_type, model->device_id);
    
    llaisysLinear(logits, last_token_state, model->weights.out_embed, model->zero_bias_voc);
    
    // Argmax
    size_t shape_scalar[1] = {1};
    llaisysTensor_t max_idx = tensorCreate(shape_scalar, 1, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
    llaisysTensor_t max_val = tensorCreate(shape_scalar, 1, model->meta.dtype, model->device_type, model->device_id);
    
    size_t shape_logits_1d[1] = {model->meta.voc};
    llaisysTensor_t logits_1d = tensorView(logits, shape_logits_1d, 1);
    
    llaisysArgmax(max_idx, max_val, logits_1d);
    
    tensorDestroy(logits_1d);

    int64_t result_token;
    void* res_ptr = tensorGetData(max_idx);
    const auto runtime = llaisysGetRuntimeAPI(model->device_type);
    runtime->memcpy_sync(&result_token, res_ptr, sizeof(int64_t), LLAISYS_MEMCPY_D2H);

    tensorDestroy(x_final);
    tensorDestroy(last_token_state);
    tensorDestroy(logits);
    tensorDestroy(max_idx);
    tensorDestroy(max_val);

    model->pos += ntoken;

    return result_token;
}

} // extern "C"
