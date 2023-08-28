#ifndef _LIB_RESIDUAL_ATTENTION_BLOCK_H_
#define _LIB_RESIDUAL_ATTENTION_BLOCK_H_

#include <vector>

#include "attention.h"
#include "layernorm.h"
#include "matmul.h"

struct ResidualCrossAttentionBlock {
  int n_max_ctx;
  int n_state;
  int n_head;
  int inner_size;
  int cross_attn_kv_size;

  std::vector<__fp16> x_copy;

  std::vector<__fp16> attn_ln_gamma;
  std::vector<__fp16> attn_ln_beta;
  CachingAttention attn;

  std::vector<__fp16> cross_attn_ln_gamma;
  std::vector<__fp16> cross_attn_ln_beta;
  ConstantKVAttention cross_attn;

  std::vector<__fp16> mlp_ln_gamma;
  std::vector<__fp16> mlp_ln_beta;

  Matmul fc1;
  std::vector<__fp16> fc1_bias;

  Matmul fc2;
  std::vector<__fp16> fc2_bias;

  std::vector<__fp16> y;

  ResidualCrossAttentionBlock(int n_max_ctx, int n_state, int n_head,
                              int cross_attn_kv_size);

  void call(int n_ctx);
  void reset();
};

struct ResidualCrossAttentionLayers {
  std::vector<ResidualCrossAttentionBlock*> blocks;

  ResidualCrossAttentionLayers(int n_layer, int n_max_ctx, int n_state,
                               int n_head, int cross_attn_kv_size);
  ~ResidualCrossAttentionLayers();

  void call(int n_ctx);
};

struct ResidualSelfAttentionBlock {
  int n_ctx;
  int n_state;
  int n_head;
  int inner_size;

  std::vector<__fp16> x_copy;

  std::vector<__fp16> attn_ln_gamma;
  std::vector<__fp16> attn_ln_beta;
  FixedShapeAttention attn;

  std::vector<__fp16> mlp_ln_gamma;
  std::vector<__fp16> mlp_ln_beta;

  Matmul fc1;
  std::vector<__fp16> fc1_bias;

  Matmul fc2;
  std::vector<__fp16> fc2_bias;

  std::vector<__fp16> y;

  ResidualSelfAttentionBlock(int n_ctx, int n_state, int n_head);

  void call();
};

struct ResidualSelfAttentionLayers {
  std::vector<ResidualSelfAttentionBlock*> blocks;

  ResidualSelfAttentionLayers(int n_layer, int n_ctx, int n_state, int n_head);
  ~ResidualSelfAttentionLayers();

  void call();
};

#endif  // _LIB_RESIDUAL_ATTENTION_BLOCK_H_
