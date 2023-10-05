#include "softmax.h"

#include <arm_neon.h>

#include <cmath>
#include <vector>

#include "misc.h"

namespace {

struct _exp_lut {
  static constexpr int LUT_SIZE = 1 << 15;
  uint16_t lut[LUT_SIZE];

  _exp_lut() {
    for (int i = 0; i < LUT_SIZE; ++i) {
      uint16_t x = i | 0x8000;
      __fp16 y = std::exp(to_f16(x));
      lut[i] = to_u16(y);
    }
  }

  __fp16 operator()(__fp16 x) const {
    uint16_t u16 = to_u16(x);
    u16 = u16 & 0x7FFF;
    return to_f16(lut[u16]);
  }
} exp_lut;

}  // namespace

void copy_C_to_fp16(Matmul* src, __fp16* dst, int rows, int cols) {
  float32_t* C = src->get_C_ptr();
  int i = 0;
  for (i = 0; i < cols - (cols % 4); i += 4) {
    // src's actual number of rows (src->M) times 4 is the base for a block of
    // columns
    float32_t* block_base = C + (src->M * 4) * (i / 4);
    for (int j = 0; j < rows; ++j) {
      auto v = vld1q_f32(block_base + j * 4);
      auto v_f16 = vcvt_f16_f32(v);
      vst1_f16(dst + j * cols + i, v_f16);
    }
  }

  if (cols % 4 != 0) {
    float32_t* block_base = C + (src->M * 4) * (i / 4);
    for (int j = 0; j < rows; ++j) {
      for (int rem_i = 0; rem_i < cols - i; ++rem_i) {
        dst[j * cols + i + rem_i] = block_base[rem_i + j * 4];
      }
    }
  }
}

__fp16 compute_max(__fp16* src, int N) {
  __fp16 max = src[0];
  int i;
  for (i = 0; i < N - (N % 8); i += 8) {
    max = std::max(max, vmaxnmvq_f16(vld1q_f16(src + i)));
  }

  // Handle any remaining values if N % 8 != 0.
  for (; i < N; ++i) {
    max = std::max(max, src[i]);
  }
  return max;
}

void log_softmax(__fp16* src, int N, __fp16 max) {
  int i;
  float sum = 0.0;
  for (i = 0; i < N - (N % 8); i += 8) {
    auto v = vld1q_f16(src + i);
    v = vsubq_f16(v, vdupq_n_f16(max));
    sum += exp_lut(vgetq_lane_f16(v, 0));
    sum += exp_lut(vgetq_lane_f16(v, 1));
    sum += exp_lut(vgetq_lane_f16(v, 2));
    sum += exp_lut(vgetq_lane_f16(v, 3));
    sum += exp_lut(vgetq_lane_f16(v, 4));
    sum += exp_lut(vgetq_lane_f16(v, 5));
    sum += exp_lut(vgetq_lane_f16(v, 6));
    sum += exp_lut(vgetq_lane_f16(v, 7));
  }
  if (N % 8 != 0) {
    for (int rem_i = 0; rem_i < N - i; ++rem_i) {
      sum += exp_lut(src[i + rem_i] - max);
    }
  }
  float log_sum = std::log(sum);
  __fp16 subtrahend = log_sum + max;
  for (i = 0; i < N - (N % 8); i += 8) {
    auto v = vld1q_f16(src + i);
    v = vsubq_f16(v, vdupq_n_f16(subtrahend));
    vst1q_f16(src + i, v);
  }
  if (N % 8 != 0) {
    for (int rem_i = 0; rem_i < N - i; ++rem_i) {
      src[i + rem_i] -= subtrahend;
    }
  }
}

void softmax_C_to_A(Matmul* src, Matmul* dst, int rows, int cols) {
  std::vector<float> row_max(rows, -std::numeric_limits<float>::infinity());
  float32_t* C = src->get_C_ptr();
  float16_t* A = dst->get_A_ptr();
  int i = 0;
  for (i = 0; i < cols - (cols % 4); i += 4) {
    // src's actual number of rows (src->M) times 4 is the base for a block of
    // columns
    float32_t* block_base = C + (src->M * 4) * (i / 4);
    for (int j = 0; j < rows; ++j) {
      auto v = vld1q_f32(block_base + j * 4);
      float32_t m = vmaxnmvq_f32(v);
      row_max[j] = std::max(row_max[j], m);
    }
  }

  if (cols % 4 != 0) {
    float32_t* block_base = C + (src->M * 4) * (i / 4);
    for (int j = 0; j < rows; ++j) {
      for (int rem_i = 0; rem_i < cols - i; ++rem_i) {
        row_max[j] = std::max(row_max[j], block_base[rem_i + j * 4]);
      }
    }
  }

  std::vector<float> row_exp_sum(rows, 0.0);

  i = 0;
  for (i = 0; i < cols - (cols % 4); i += 4) {
    float32_t* block_base = C + (src->M * 4) * (i / 4);
    float16_t* dst_block_base = A + (dst->M * 8) * (i / 8);
    for (int j = 0; j < rows; ++j) {
      auto v = vld1q_f32(block_base + j * 4);
      v = vsubq_f32(v, vdupq_n_f32(row_max[j]));
      auto v_f16 = vcvt_f16_f32(v);
      v_f16 = vset_lane_f16(exp_lut(vget_lane_f16(v_f16, 0)), v_f16, 0);
      v_f16 = vset_lane_f16(exp_lut(vget_lane_f16(v_f16, 1)), v_f16, 1);
      v_f16 = vset_lane_f16(exp_lut(vget_lane_f16(v_f16, 2)), v_f16, 2);
      v_f16 = vset_lane_f16(exp_lut(vget_lane_f16(v_f16, 3)), v_f16, 3);
      row_exp_sum[j] += vaddvq_f32(vcvt_f32_f16(v_f16));
      vst1_f16(dst_block_base + j * 8 + ((i % 8) / 4) * 4, v_f16);
    }
  }

  if (cols % 4 != 0) {
    float32_t* block_base = C + (src->M * 4) * (i / 4);
    float16_t* dst_block_base = A + (dst->M * 8) * (i / 8);
    for (int j = 0; j < rows; ++j) {
      for (int rem_i = 0; rem_i < cols - i; ++rem_i) {
        __fp16 elem = block_base[rem_i + j * 4] - row_max[j];
        elem = exp_lut(elem);
        row_exp_sum[j] += elem;
        dst_block_base[j * 8 + ((i % 8) / 4) * 4 + rem_i] = elem;
      }
    }
  }

  for (i = 0; i < cols - (cols % 8); i += 8) {
    float16_t* block_base = A + (dst->M * 8) * (i / 8);
    for (int j = 0; j < rows; ++j) {
      auto v = vld1q_f16(block_base + j * 8);
      v = vdivq_f16(v, vdupq_n_f16((__fp16)row_exp_sum[j]));
      vst1q_f16(block_base + j * 8, v);
    }
  }
  if (cols % 8 != 0) {
    float16_t* block_base = A + (dst->M * 8) * (i / 8);
    for (int j = 0; j < rows; ++j) {
      for (int rem_i = 0; rem_i < cols - i; ++rem_i) {
        block_base[rem_i + j * 8] /= (__fp16)row_exp_sum[j];
      }
    }
  }
}
