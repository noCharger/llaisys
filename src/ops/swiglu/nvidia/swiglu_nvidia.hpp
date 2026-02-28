#pragma once
#include "llaisys/tensor.h"

namespace llaisys::ops::nvidia {
void swiglu(tensor_t out, tensor_t gate, tensor_t up);
}
