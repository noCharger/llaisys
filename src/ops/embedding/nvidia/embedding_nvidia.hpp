#pragma once
#include "llaisys/tensor.h"

namespace llaisys::ops::nvidia {
void embedding(tensor_t out, tensor_t index, tensor_t weight);
}
