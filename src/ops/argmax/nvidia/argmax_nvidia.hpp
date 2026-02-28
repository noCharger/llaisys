#pragma once
#include "llaisys/tensor.h"

namespace llaisys::ops::nvidia {
void argmax(tensor_t max_val, tensor_t max_idx, tensor_t in);
}
