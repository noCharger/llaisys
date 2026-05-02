#pragma once

#include "../../tensor/tensor.hpp"

#include <cstdint>
#include <vector>

namespace llaisys::ops {

// Variable-length self-attention. cu_seqlens_q (length batch+1) marks
// boundaries inside packed q/attn_val; each request reads from its own
// k_blocks[r] / v_blocks[r] (caller slices to the valid prefix).
// Causal mask: query i within request r attends to k [0, k_len - q_len + i].
void self_attention_varlen(tensor_t attn_val,
                           tensor_t q,
                           const std::vector<tensor_t> &k_blocks,
                           const std::vector<tensor_t> &v_blocks,
                           const int32_t *cu_seqlens_q,
                           int32_t batch,
                           float scale);

} // namespace llaisys::ops
