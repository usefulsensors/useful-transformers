#include "residual_attention_block.h"

#include <cassert>
#include <cmath>
#include <cstring>

#include "bias.h"

ResidualCrossAttentionBlock::ResidualCrossAttentionBlock(int n_max_ctx,
                                                         int n_state,
                                                         int n_head,
                                                         int cross_attn_kv_size)
    : n_max_ctx(n_max_ctx),
      n_state(n_state),
      n_head(n_head),
      inner_size(n_state / n_head),
      cross_attn_kv_size(cross_attn_kv_size),
      x_copy(n_max_ctx * n_state),
      attn_ln_gamma(n_state),
      attn_ln_beta(n_state),
      attn(n_max_ctx, n_state, n_head),
      cross_attn_ln_gamma(n_state),
      cross_attn_ln_beta(n_state),
      cross_attn(n_max_ctx, n_state, n_head, cross_attn_kv_size),
      mlp_ln_gamma(n_state),
      mlp_ln_beta(n_state),
      fc1(n_max_ctx, n_state, n_state * 4),
      fc1_bias(n_state * 4),
      fc2(n_max_ctx, n_state * 4, n_state),
      fc2_bias(n_state),
      y(n_max_ctx * n_state) {}


void ResidualCrossAttentionBlock::call(int n_ctx) {
  assert(n_ctx <= n_max_ctx && "n_ctx > n_max_ctx");

  memcpy(x_copy.data(), attn.Q.get_A_ptr(), n_ctx * n_state * sizeof(__fp16));

  layernorm_A(&attn.Q, n_ctx, n_state, attn_ln_gamma, attn_ln_beta);
  attn.call(n_ctx);

  add_residual_C_to_A(&attn.out, &cross_attn.Q, x_copy, n_ctx, n_state);

  memcpy(x_copy.data(), cross_attn.Q.get_A_ptr(),
         n_ctx * n_state * sizeof(__fp16));

  layernorm_A(&cross_attn.Q, n_ctx, n_state, cross_attn_ln_gamma,
              cross_attn_ln_beta);
  cross_attn.call(n_ctx);

  add_residual_C_to_A(&cross_attn.out, &fc1, x_copy, n_ctx, n_state);

  memcpy(x_copy.data(), fc1.get_A_ptr(), n_ctx * n_state * sizeof(__fp16));

  layernorm_A(&fc1, n_ctx, n_state, mlp_ln_gamma, mlp_ln_beta);
  fc1.call();

  bias_and_gelu_C_to_A(&fc1, &fc2, fc1_bias, n_ctx, n_state * 4);

  fc2.call();

  bias_and_add_residual_C(&fc2, y, x_copy, fc2_bias, n_ctx, n_state);
}

void ResidualCrossAttentionBlock::reset() {
  attn.reset();
  cross_attn.reset();
}

ResidualCrossAttentionLayers::ResidualCrossAttentionLayers(
    int n_layer, int n_max_ctx, int n_state, int n_head, int cross_attn_kv_size)
    : blocks(n_layer) {
  for (int i = 0; i < n_layer; ++i) {
    blocks[i] = new ResidualCrossAttentionBlock(n_max_ctx, n_state, n_head,
                                                cross_attn_kv_size);
  }
}

ResidualCrossAttentionLayers::~ResidualCrossAttentionLayers() {
  for (int i = 0; i < blocks.size(); ++i) {
    delete blocks[i];
  }
}

void ResidualCrossAttentionLayers::call(int n_ctx) {
  int num_blocks = blocks.size();
  int n_state = blocks[0]->n_state;
  blocks[0]->call(n_ctx);

  for (int block = 1; block < num_blocks; ++block) {
    blocks[block]->attn.Q.set_A(blocks[block - 1]->y.data(), n_ctx);
    blocks[block]->call(n_ctx);
  }
}

ResidualSelfAttentionBlock::ResidualSelfAttentionBlock(int n_ctx, int n_state,
                                                       int n_head)
    : n_ctx(n_ctx),
      n_state(n_state),
      n_head(n_head),
      inner_size(n_state / n_head),
      x_copy(n_ctx * n_state),
      attn_ln_gamma(n_state),
      attn_ln_beta(n_state),
      attn(n_ctx, n_state, n_head),
      mlp_ln_gamma(n_state),
      mlp_ln_beta(n_state),
      fc1(n_ctx, n_state, n_state * 4),
      fc1_bias(n_state * 4),
      fc2(n_ctx, n_state * 4, n_state),
      fc2_bias(n_state),
      y(n_ctx * n_state) {}

void ResidualSelfAttentionBlock::call() {
  memcpy(x_copy.data(), attn.Q.get_A_ptr(), n_ctx * n_state * sizeof(__fp16));

  layernorm_A(&attn.Q, n_ctx, n_state, attn_ln_gamma, attn_ln_beta);
  attn.call();

  add_residual_C_to_A(&attn.out, &fc1, x_copy, n_ctx, n_state);

  memcpy(x_copy.data(), fc1.get_A_ptr(), n_ctx * n_state * sizeof(__fp16));

  layernorm_A(&fc1, n_ctx, n_state, mlp_ln_gamma, mlp_ln_beta);
  fc1.call();

  bias_and_gelu_C_to_A(&fc1, &fc2, fc1_bias, n_ctx, n_state * 4);

  fc2.call();

  bias_and_add_residual_C(&fc2, y, x_copy, fc2_bias, n_ctx, n_state);
}

ResidualSelfAttentionLayers::ResidualSelfAttentionLayers(int n_layer, int n_ctx,
                                                         int n_state,
                                                         int n_head)
    : blocks(n_layer) {
  for (int i = 0; i < n_layer; ++i) {
    blocks[i] = new ResidualSelfAttentionBlock(n_ctx, n_state, n_head);
  }
}

ResidualSelfAttentionLayers::~ResidualSelfAttentionLayers() {
  for (int i = 0; i < blocks.size(); ++i) {
    delete blocks[i];
  }
}

void ResidualSelfAttentionLayers::call() {
  int num_blocks = blocks.size();
  int n_state = blocks[0]->n_state;
  blocks[0]->call();

  for (int block = 1; block < num_blocks; ++block) {
    blocks[block]->attn.Q.set_A(blocks[block - 1]->y.data());
    blocks[block]->call();
  }
}
