#include "whisper.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "bias.h"
#include "softmax.h"

AudioEncoder::AudioEncoder(int num_layers, int n_ctx, int n_state, int n_head)
    : num_layers(num_layers),
      n_ctx(n_ctx),
      n_state(n_state),
      n_head(n_head),
      conv0(n_ctx * 2, 80, n_state, 1),
      conv1(n_ctx * 2, n_state, n_state, 2),
      conv0_bias(n_state),
      conv1_bias(n_state),
      positional_embedding(n_ctx * n_state),
      blocks(num_layers, n_ctx, n_state, n_head),
      ln_post_gamma(n_state),
      ln_post_beta(n_state) {}

template <typename T>
void layernorm(std::vector<T> &x, int rows, int cols,
               const std::vector<T> &gamma, const std::vector<T> &beta,
               T eps = 1e-5) {
#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    float mean = x[i * cols + 0];
    for (int j = 1; j < cols; ++j) {
      mean += x[i * cols + j];
    }
    mean /= cols;
    float var = 0.0;
    for (int j = 0; j < cols; ++j) {
      float elem = x[i * cols + j] - mean;
      elem *= elem;
      var += elem;
    }
    var /= cols;
    float denom = std::sqrt(var + eps);
    for (int j = 0; j < cols; ++j) {
      float elem = x[i * cols + j];
      x[i * cols + j] = (elem - mean) / denom * gamma[j] + beta[j];
    }
  }
}

void AudioEncoder::call() {
  conv0.call();
  std::vector<__fp16> conv1_input(n_ctx * 2 * n_state);
  bias_and_gelu(conv0.output.data(), conv1_input.data(), conv0_bias, n_ctx * 2,
                n_state);

  conv1.copy_A(conv1_input.data());
  conv1.call();
  bias_and_gelu(conv1.output.data(), blocks.blocks[0]->attn.Q.get_A_ptr(),
                conv1_bias, n_ctx, n_state);

  for (int i = 0; i < n_ctx; ++i) {
    for (int j = 0; j < n_state; ++j) {
      blocks.blocks[0]->attn.Q.A_at(i, j) +=
          positional_embedding[i * n_state + j];
    }
  }
  blocks.call();
  int last_layer_idx = blocks.blocks.size() - 1;
  layernorm(blocks.blocks[last_layer_idx]->y, n_ctx, n_state, ln_post_gamma,
            ln_post_beta);
}

TextDecoder::TextDecoder(int num_layers, int n_text_max_ctx, int n_state,
                         int n_head, int n_audio_ctx, int n_vocab)
    : num_layers(num_layers),
      n_text_max_ctx(n_text_max_ctx),
      n_state(n_state),
      n_head(n_head),
      n_audio_ctx(n_audio_ctx),
      n_vocab(n_vocab),
      positional_embedding(n_text_max_ctx * n_state),
      blocks(num_layers, 1, n_state, n_head, n_audio_ctx),
      ln_gamma(n_state),
      ln_beta(n_state),
      detokenizer0(1, n_state, n_vocab / 3, 0),
      detokenizer1(1, n_state, n_vocab / 3, 1),
      detokenizer2(1, n_state, n_vocab - 2 * n_vocab / 3, 2) {
}

void TextDecoder::call(int prompt) {
  int offset = blocks.blocks[0]->attn.cur_kv_len;
  int slice_len = n_vocab / 3;
  int last_slice_len = n_vocab - 2 * slice_len;
  Matmul *detokenizer = prompt < slice_len       ? &detokenizer0
                        : prompt < 2 * slice_len ? &detokenizer1
                                                 : &detokenizer2;
  for (int j = 0; j < n_state; ++j) {
    int col = (prompt < 2 * slice_len) ? (prompt % slice_len) : ((prompt - 2 * slice_len) % last_slice_len);
    blocks.blocks[0]->attn.Q.A_at(0, j) =
        detokenizer->B_at(j, col) +
        positional_embedding[(0 + offset) * n_state + j];
  }
  blocks.call(1);
  int last_layer_idx = blocks.blocks.size() - 1;
  layernorm(blocks.blocks[last_layer_idx]->y, 1, n_state, ln_gamma, ln_beta);

#pragma omp parallel sections
  {
#pragma omp section
    {
      detokenizer0.set_A(blocks.blocks[last_layer_idx]->y.data());
      detokenizer0.call();
    }
#pragma omp section
    {
      detokenizer1.set_A(blocks.blocks[last_layer_idx]->y.data());
      detokenizer1.call();
    }
#pragma omp section
    {
      detokenizer2.set_A(blocks.blocks[last_layer_idx]->y.data());
      detokenizer2.call();
    }
  }
}

inline void suppress(__fp16 *src, int begin, int end, int size,
                     const std::vector<int> &tokens) {
  __fp16 minus_inf = -std::numeric_limits<__fp16>::infinity();
  for (int i = 0; i < tokens.size(); ++i) {
    int token = tokens[i];
    if (token < begin || token >= end) continue;
    src[token % size] = minus_inf;
  }
}

void TextDecoder::get_logits(__fp16 *logits) {
#pragma omp parallel sections
  {
#pragma omp section
    { copy_C_to_fp16(&detokenizer0, logits, 1, n_vocab / 3); }
#pragma omp section
    { copy_C_to_fp16(&detokenizer1, logits + n_vocab / 3, 1, n_vocab / 3); }
#pragma omp section
    { copy_C_to_fp16(&detokenizer2, logits + 2 * n_vocab / 3, 1, n_vocab - 2 * n_vocab / 3); }
  }
}

void TextDecoder::log_softmax(__fp16 *logits,
                              const std::vector<int> &suppress_tokens) {
  __fp16 max0, max1, max2;
#pragma omp parallel sections
  {
#pragma omp section
    {
      copy_C_to_fp16(&detokenizer0, logits, 1, n_vocab / 3);
      suppress(logits, 0, n_vocab / 3, n_vocab / 3, suppress_tokens);
      max0 = compute_max(logits, n_vocab / 3);
    }
#pragma omp section
    {
      copy_C_to_fp16(&detokenizer1, logits + n_vocab / 3, 1, n_vocab / 3);
      suppress(logits + n_vocab / 3, n_vocab / 3, 2 * n_vocab / 3, n_vocab / 3,
               suppress_tokens);
      max1 = compute_max(logits + n_vocab / 3, n_vocab / 3);
    }
#pragma omp section
    {
      copy_C_to_fp16(&detokenizer2, logits + 2 * n_vocab / 3, 1, n_vocab - 2 * n_vocab / 3);
      suppress(logits + 2 * n_vocab / 3, 2 * n_vocab / 3, n_vocab, n_vocab - 2 * n_vocab / 3,
               suppress_tokens);
      max2 = compute_max(logits + 2 * n_vocab / 3, n_vocab - 2 * n_vocab / 3);
    }
  }
  __fp16 max = std::max(std::max(max0, max1), max2);
  ::log_softmax(logits, n_vocab, max);
}

WhisperModel::WhisperModel(int n_mels, int n_audio_ctx, int n_audio_state,
                           int n_audio_head, int n_audio_layer, int n_text_ctx,
                           int n_text_state, int n_text_head, int n_text_layer,
                           int n_vocab)
    : n_mels(n_mels),
      n_audio_ctx(n_audio_ctx),
      n_audio_state(n_audio_state),
      n_audio_head(n_audio_head),
      n_audio_layer(n_audio_layer),
      n_text_ctx(n_text_ctx),
      n_text_state(n_text_state),
      n_text_head(n_text_head),
      n_text_layer(n_text_layer),
      n_vocab(n_vocab),
      encoder(n_audio_layer, n_audio_ctx, n_audio_state, n_audio_head),
      decoder(n_text_layer, n_text_ctx, n_text_state, n_text_head, n_audio_ctx,
              n_vocab) {}
