#ifndef _LIB_ATTENTION_H_
#define _LIB_ATTENTION_H_

#include <vector>

#include "matmul.h"

// 'x's shape can be [n_ctx, n_state] where 0 < n_ctx <= n_max_ctx. n_max_ctx is
// fixed during construction time. 'k' and 'v' keep growing on successive calls.
// They have to be explicitly reset by calling 'reset()'.
struct CachingAttention {
  int n_max_ctx;
  int n_state;
  int n_head;
  int inner_size;
  int cur_kv_len;
  int cur_kv_size;

  Matmul Q;
  Matmul K;
  Matmul V;

  std::vector<__fp16> q_bias;
  std::vector<__fp16> v_bias;

  std::vector<Matmul*> outer_product;

  // The coefficients for the rows of v
  std::vector<float> w;

  std::vector<Matmul*> wv;

  Matmul out;
  std::vector<__fp16> out_bias;

  CachingAttention(int n_max_ctx, int n_state, int n_head);
  ~CachingAttention();
  void call(int n_ctx);
  void reset();
};

// 'k' and 'v' are constant matrices of size [kv_size, n_state]. They can be set
// by calling 'reset' after setting the A operand of K to a [kv_size, n_state]
// matrix.
struct ConstantKVAttention {
  int n_max_ctx;
  int n_state;
  int n_head;
  int inner_size;
  int kv_size;

  Matmul Q;
  Matmul K;
  Matmul V;

  std::vector<__fp16> q_bias;
  std::vector<__fp16> v_bias;

  std::vector<Matmul*> outer_product;

  std::vector<Matmul*> wv;

  Matmul out;
  std::vector<__fp16> out_bias;

  ConstantKVAttention(int n_max_ctx, int n_state, int n_head, int kv_size);
  ~ConstantKVAttention();
  void call(int n_ctx);
  void reset();
};

// 'q', 'k', and 'v' are the same shape, same as the incoming 'x' on every call.
struct FixedShapeAttention {
  int n_ctx;
  int n_state;
  int n_head;
  int inner_size;
  Matmul Q;
  Matmul K;
  Matmul V;

  std::vector<__fp16> q_bias;
  std::vector<__fp16> v_bias;

  std::vector<Matmul*> outer_product;
  std::vector<Matmul*> wv;

  Matmul out;
  std::vector<__fp16> out_bias;

  FixedShapeAttention(int n_ctx, int n_state, int n_head);
  ~FixedShapeAttention();
  void call();
};

#endif  // _LIB_ATTENTION_H_
