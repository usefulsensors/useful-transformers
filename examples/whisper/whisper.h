#ifndef _EXAMPLES_WHISPER_WHISPER_H_
#define _EXAMPLES_WHISPER_WHISPER_H_

#include <vector>

#include "conv.h"
#include "matmul.h"
#include "residual_attention_block.h"

struct AudioEncoder {
  int num_layers;
  int n_ctx;
  int n_state;
  int n_head;

  Conv2D1x3 conv0;
  Conv2D1x3 conv1;

  std::vector<__fp16> conv0_bias;
  std::vector<__fp16> conv1_bias;

  std::vector<__fp16> positional_embedding;

  ResidualSelfAttentionLayers blocks;

  std::vector<__fp16> ln_post_gamma;
  std::vector<__fp16> ln_post_beta;

  AudioEncoder(int num_layers, int n_ctx, int n_state, int n_head);
  void call();
};

struct TextDecoder {
  int num_layers;
  int n_text_max_ctx;
  int n_state;
  int n_head;
  int n_audio_ctx;
  int n_vocab;

  std::vector<__fp16> positional_embedding;

  ResidualCrossAttentionLayers blocks;

  std::vector<__fp16> ln_gamma;
  std::vector<__fp16> ln_beta;

  Matmul detokenizer0;
  Matmul detokenizer1;
  Matmul detokenizer2;

  TextDecoder(int num_layers, int n_text_max_ctx, int n_state, int n_head,
              int n_audio_ctx, int n_vocab);
  void call(int prompt);
  void get_logits(__fp16* logits);
  void log_softmax(__fp16* dst, const std::vector<int>& suppress_tokens);
};

struct WhisperModel {
  int n_mels;
  int n_audio_ctx;
  int n_audio_state;
  int n_audio_head;
  int n_audio_layer;
  int n_text_ctx;
  int n_text_state;
  int n_text_head;
  int n_text_layer;
  int n_vocab;

  AudioEncoder encoder;
  TextDecoder decoder;

  WhisperModel(int n_mels, int n_audio_ctx, int n_audio_state, int n_audio_head,
               int n_audio_layer, int n_text_ctx, int n_text_state,
               int n_text_head, int n_text_layer, int n_vocab);
};

#endif  // _EXAMPLES_WHISPER_WHISPER_H_
