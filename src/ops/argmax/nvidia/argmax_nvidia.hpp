#pragma once
#include "tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void argmax(tensor_t max_val, tensor_t max_idx, tensor_t in);
}
