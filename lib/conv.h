#ifndef _LIB_CONV_H_
#define _LIB_CONV_H_

#include <arm_neon.h>

#include <vector>

#include "matmul.h"

/*
 * Conv2D with 1x3xD weights.
 *
 * Each of the three weight vectors is treated as a separate matmul, The outputs
 * are then added together, shifted by the relative position of each weight
 * vector.
 */
struct Conv2D1x3 {
  int input_width;
  int output_width;
  int input_depth;
  int output_depth;
  int stride;

  std::vector<__fp16> bias;
  std::vector<float> output;

  Matmul channel0;
  Matmul channel1;
  Matmul channel2;

  Conv2D1x3(int input_width, int input_depth, int output_depth, int stride);

  void copy_weights(const float* weights);
  void copy_weights(const __fp16* weights);
  void copy_A(const float* input);
  void copy_A(const __fp16* input);
  void call();
};

#endif  // _LIB_CONV_H_
