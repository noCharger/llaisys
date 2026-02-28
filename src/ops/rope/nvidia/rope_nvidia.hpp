#pragma once
#include "llaisys/tensor.h"

namespace llaisys::ops::nvidia {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
}
