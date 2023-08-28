#ifndef _LIB_BIAS_H_
#define _LIB_BIAS_H_

#include <vector>

#include "matmul.h"

void bias_and_gelu_C_to_A(Matmul* src, Matmul* dst,
                          const std::vector<__fp16>& bias, int rows, int cols);

void bias_and_gelu(float* src, __fp16* dst, const std::vector<__fp16>& bias,
                   int rows, int cols);

void bias_and_scale_C(Matmul* dst, const std::vector<__fp16>& bias, float scale,
                      int rows, int cols);

void add_residual_C_to_A(Matmul* src, Matmul* dst, std::vector<__fp16> residual,
                         int rows, int cols);

void bias_and_add_residual_C(Matmul* src, std::vector<__fp16>& dst,
                             std::vector<__fp16> residual,
                             std::vector<__fp16> bias, int rows, int cols);

#endif  // _LIB_BIAS_H_
