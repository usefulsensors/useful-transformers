#ifndef _LIB_LAYERNORM_H_
#define _LIB_LAYERNORM_H_

#include <vector>

#include "matmul.h"

void layernorm_A(Matmul *x, int rows, int cols,
                 const std::vector<__fp16> &gamma,
                 const std::vector<__fp16> &beta, __fp16 eps = 1e-5);

#endif  // _LIB_LAYERNORM_H_
