#pragma once
#include "llaisys/tensor.h"

namespace llaisys::ops::nvidia {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
}
