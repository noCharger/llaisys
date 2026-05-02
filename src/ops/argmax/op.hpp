#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);

// Per-row argmax. out_indices [N] i64, logits [N, voc]. Greedy fallback
// for batched inference; max values are not returned (not needed by the
// scheduler hot path).
void argmax_batch(tensor_t out_indices, tensor_t logits);
}
