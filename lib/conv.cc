#include "conv.h"

#include <arm_neon.h>

#include <cstring>
#include <vector>

#include "matview.h"

namespace {

void sum_results_add_bias(float* result, float* out0, float* out1, float* out2,
                          __fp16* bias, int out_width, int out_depth) {
  float* result_ptr = result;
  float* out_ptr0 = out0;
  float* out_ptr1 = out1;
  float* out_ptr2 = out2;

  for (int i = 0; i < out_depth; i += 4) {
    float32x4_t B = vcvt_f32_f16(vld1_f16(&bias[i]));
    for (int j = out_width - 1; j >= 0; j--) {
      float32x4_t O0 = vld1q_f32(out_ptr0);
      float32x4_t O1 = vld1q_f32(out_ptr1);
      float32x4_t O2 = vld1q_f32(out_ptr2);

      float32x4_t accum = vaddq_f32(O0, O1);
      accum = vaddq_f32(accum, O2);
      accum = vaddq_f32(accum, B);

      vst1q_f32(result_ptr, accum);

      result_ptr += 4;
      out_ptr0 += 4;
      out_ptr1 += 4;
      out_ptr2 += 4;
    }
  }
}

}  // namespace.

Conv2D1x3::Conv2D1x3(int input_width, int input_depth, int output_depth,
                     int stride)
    : input_width(input_width),
      output_width(input_width / stride),
      input_depth(input_depth),
      output_depth(output_depth),
      stride(stride),
      output(output_width * output_depth),
      bias(output_depth),
      channel0(output_width, input_depth, output_depth),
      channel1(output_width, input_depth, output_depth),
      channel2(output_width, input_depth, output_depth) {}

void Conv2D1x3::copy_weights(const float* weights) {
  std::vector<__fp16> w0(input_depth * output_depth);
  std::vector<__fp16> w1(input_depth * output_depth);
  std::vector<__fp16> w2(input_depth * output_depth);

  MatView weights_view(weights, output_depth, 3, input_depth);

  for (int i = 0; i < output_depth; i++) {
    for (int j = 0; j < input_depth; j++) {
      int out_idx = j * output_depth + i;
      w0[out_idx] = weights_view(i, 0, j);
      w1[out_idx] = weights_view(i, 1, j);
      w2[out_idx] = weights_view(i, 2, j);
    }
  }

  channel0.set_B(&w0[0]);
  channel1.set_B(&w1[0]);
  channel2.set_B(&w2[0]);
}

void Conv2D1x3::copy_weights(const __fp16* weights) {
  std::vector<__fp16> w0(input_depth * output_depth);
  std::vector<__fp16> w1(input_depth * output_depth);
  std::vector<__fp16> w2(input_depth * output_depth);

  MatView weights_view(weights, output_depth, 3, input_depth);

  for (int i = 0; i < output_depth; i++) {
    for (int j = 0; j < input_depth; j++) {
      int out_idx = j * output_depth + i;
      w0[out_idx] = weights_view(i, 0, j);
      w1[out_idx] = weights_view(i, 1, j);
      w2[out_idx] = weights_view(i, 2, j);
    }
  }

  channel0.set_B(&w0[0]);
  channel1.set_B(&w1[0]);
  channel2.set_B(&w2[0]);
}

// Floating point input assumes stride of 1 and native layout.
void Conv2D1x3::copy_A(const float* input) {
  std::vector<float> in0(input_width * input_depth, 0);
  std::vector<float> in1(input_width * input_depth);
  std::vector<float> in2(input_width * input_depth, 0);

  int copy_len = input_depth * sizeof(float);
#pragma omp parallel for
  for (int i = 0; i < input_width - 1; i++) {
    int idx = i * input_depth;
    memcpy(&in0[idx + input_depth], &input[idx], copy_len);
    memcpy(&in1[idx], &input[idx], copy_len);
    memcpy(&in2[idx], &input[idx + input_depth], copy_len);
  }
  memcpy(&in1[(input_width - 1) * input_depth],
         &input[(input_width - 1) * input_depth], copy_len);

  channel0.set_A(&in0[0]);
  channel1.set_A(&in1[0]);
  channel2.set_A(&in2[0]);
}

// fp16 input assumes a stride of two and perf layout.
void Conv2D1x3::copy_A(const __fp16* input) {
  MatView input_view(input, input_depth / 8, input_width, 8);

#pragma omp parallel for
  for (int i = 0; i < input_width / stride; i++) {
    for (int j = 0; j < input_depth; j++) {
      channel0.A_at(i, j) =
          i == 0 ? 0 : input_view(j / 8, i * stride - 1, j % 8);
      channel1.A_at(i, j) = input_view(j / 8, i * stride, j % 8);
      channel2.A_at(i, j) = input_view(j / 8, i * stride + 1, j % 8);
    }
  }
}

void Conv2D1x3::call() {
  channel0.call();
  channel1.call();
  channel2.call();

  // Add resulting vectors together.
  float* out0 = channel0.get_C_ptr();
  float* out1 = channel1.get_C_ptr();
  float* out2 = channel2.get_C_ptr();
  sum_results_add_bias(&output[0], out0, out1, out2, &bias[0], output_width,
                       output_depth);
}
