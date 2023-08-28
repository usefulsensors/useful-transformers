#ifndef _LIB_SOFTMAX_H_
#define _LIB_SOFTMAX_H_

#include <vector>

#include "matmul.h"

void softmax_C_to_A(Matmul* src, Matmul* dst, int rows, int cols);

// Copy from C operand of src to dst
void copy_C_to_fp16(Matmul* src, __fp16* dst, int rows, int cols);
__fp16 compute_max(__fp16* src, int N);
void log_softmax(__fp16* src, int N, __fp16 max);

#endif  // _LIB_SOFTMAX_H_
