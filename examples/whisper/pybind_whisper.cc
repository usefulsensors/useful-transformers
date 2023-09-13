#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "whisper.h"

namespace py = pybind11;

#define XSTR(s) STR(s)
#define STR(s) #s

#define SET_MATRIX_LAYER_PREFIX(ClassName, m, M, prefix)                     \
  def(STR(set_##m),                                                          \
      [](ClassName &self, int i, py::array_t<float, py::array::c_style> W) { \
        const float *data = static_cast<const float *>(W.request().ptr);     \
        self.prefix.blocks.blocks[i]->M.set_B(data);                         \
      })

#define SET_MATRIX_LAYER(ClassName, m, M)                                    \
  def(STR(set_##m),                                                          \
      [](ClassName &self, int i, py::array_t<float, py::array::c_style> W) { \
        const float *data = static_cast<const float *>(W.request().ptr);     \
        self.blocks.blocks[i]->M.set_B(data);                                \
      })

#define SET_CONV_DIRECT(ClassName, m, M)                                 \
  def(STR(set_##m),                                                      \
      [](ClassName &self, py::array_t<float, py::array::c_style> W) {    \
        const float *data = static_cast<const float *>(W.request().ptr); \
        self.M.copy_weights(data);                                       \
      })

#define SET_VECTOR_LAYER_PREFIX(ClassName, v, V, prefix)                      \
  def(STR(set_##v),                                                           \
      [](ClassName &obj, int l, py::array_t<float, py::array::c_style> vec) { \
        const float *data = static_cast<const float *>(vec.request().ptr);    \
        for (int i = 0; i < obj.prefix.blocks.blocks[l]->V.size(); ++i)       \
          obj.prefix.blocks.blocks[l]->V[i] = data[i];                        \
      })

#define SET_VECTOR_LAYER(ClassName, v, V)                                     \
  def(STR(set_##v),                                                           \
      [](ClassName &obj, int l, py::array_t<float, py::array::c_style> vec) { \
        const float *data = static_cast<const float *>(vec.request().ptr);    \
        for (int i = 0; i < obj.blocks.blocks[l]->V.size(); ++i)              \
          obj.blocks.blocks[l]->V[i] = data[i];                               \
      })

#define SET_MATRIX_DIRECT(ClassName, m, M)                               \
  def(STR(set_##m),                                                      \
      [](ClassName &obj, py::array_t<float, py::array::c_style> W) {     \
        const float *data = static_cast<const float *>(W.request().ptr); \
        obj.M.set_B(data);                                               \
      })

#define SET_VECTOR_DIRECT(ClassName, v, V)                                 \
  def(STR(set_##v),                                                        \
      [](ClassName &obj, py::array_t<float, py::array::c_style> vec) {     \
        const float *data = static_cast<const float *>(vec.request().ptr); \
        for (int i = 0; i < obj.V.size(); ++i) obj.V[i] = data[i];         \
      })

PYBIND11_MODULE(pybind_whisper, m) {
  m.doc() = "Whisper speech to text converter";
  using numpy_float_array = py::array_t<float, py::array::c_style>;
  using numpy_uint16_array = py::array_t<uint16_t, py::array::c_style>;

  py::class_<AudioEncoder>(m, "AudioEncoder")
      .def(py::init<int, int, int, int>())
      .SET_MATRIX_LAYER(AudioEncoder, attn_Wq, attn.Q)
      .SET_MATRIX_LAYER(AudioEncoder, attn_Wk, attn.K)
      .SET_MATRIX_LAYER(AudioEncoder, attn_Wv, attn.V)
      .SET_MATRIX_LAYER(AudioEncoder, attn_Wout, attn.out)
      .SET_MATRIX_LAYER(AudioEncoder, Wfc1, fc1)
      .SET_MATRIX_LAYER(AudioEncoder, Wfc2, fc2)
      .SET_VECTOR_LAYER(AudioEncoder, attn_out_bias, attn.out_bias)
      .SET_VECTOR_LAYER(AudioEncoder, attn_q_bias, attn.q_bias)
      .SET_VECTOR_LAYER(AudioEncoder, attn_v_bias, attn.v_bias)
      .SET_VECTOR_LAYER(AudioEncoder, fc1_bias, fc1_bias)
      .SET_VECTOR_LAYER(AudioEncoder, fc2_bias, fc2_bias)
      .SET_VECTOR_LAYER(AudioEncoder, mlp_ln_gamma, mlp_ln_gamma)
      .SET_VECTOR_LAYER(AudioEncoder, mlp_ln_beta, mlp_ln_beta)
      .SET_VECTOR_LAYER(AudioEncoder, attn_ln_gamma, attn_ln_gamma)
      .SET_VECTOR_LAYER(AudioEncoder, attn_ln_beta, attn_ln_beta)
      .SET_VECTOR_DIRECT(AudioEncoder, positional_embedding,
                         positional_embedding)
      .SET_CONV_DIRECT(AudioEncoder, conv0_weights, conv0)
      .SET_VECTOR_DIRECT(AudioEncoder, conv0_bias, conv0_bias)
      .SET_CONV_DIRECT(AudioEncoder, conv1_weights, conv1)
      .SET_VECTOR_DIRECT(AudioEncoder, conv1_bias, conv1_bias)
      .SET_VECTOR_DIRECT(AudioEncoder, ln_post_gamma, ln_post_gamma)
      .SET_VECTOR_DIRECT(AudioEncoder, ln_post_beta, ln_post_beta)
      .def("call",
           [](AudioEncoder &enc, numpy_float_array x) -> numpy_float_array {
             auto ndim = x.ndim();
             auto shape = x.shape();
             assert((ndim == 2 || (ndim == 3 && shape[0] == 1)) &&
                    "Should be 2 dimensional or batch_size should be 1");
             const float *data = static_cast<const float *>(x.request().ptr);
             int n_ctx = ndim == 2 ? shape[0] : shape[1];
             int n_state = ndim == 2 ? shape[1] : shape[2];
             assert(n_ctx / 2 == enc.n_ctx && "n_ctx should match");
             assert(n_state == 80 && "input depth should match whisper");

             enc.conv0.copy_A(data);
             enc.call();
             int last_layer_idx = enc.blocks.blocks.size() - 1;
             int sz = enc.n_ctx * enc.n_state;
             auto y = numpy_float_array(sz);
             float *y_ptr = static_cast<float *>(y.request().ptr);
             for (int i = 0; i < sz; ++i)
               y_ptr[i] = enc.blocks.blocks[last_layer_idx]->y[i];
             return y.reshape({enc.n_ctx, enc.n_state});
           });
  py::class_<TextDecoder>(m, "TextDecoder")
      .def(py::init<int, int, int, int, int, int>())
      .SET_MATRIX_LAYER(TextDecoder, attn_Wq, attn.Q)
      .SET_MATRIX_LAYER(TextDecoder, attn_Wk, attn.K)
      .SET_MATRIX_LAYER(TextDecoder, attn_Wv, attn.V)
      .SET_MATRIX_LAYER(TextDecoder, attn_Wout, attn.out)
      .SET_MATRIX_LAYER(TextDecoder, Wfc1, fc1)
      .SET_MATRIX_LAYER(TextDecoder, Wfc2, fc2)
      .SET_VECTOR_LAYER(TextDecoder, attn_out_bias, attn.out_bias)
      .SET_VECTOR_LAYER(TextDecoder, attn_q_bias, attn.q_bias)
      .SET_VECTOR_LAYER(TextDecoder, attn_v_bias, attn.v_bias)
      .SET_VECTOR_LAYER(TextDecoder, fc1_bias, fc1_bias)
      .SET_VECTOR_LAYER(TextDecoder, fc2_bias, fc2_bias)
      .SET_MATRIX_LAYER(TextDecoder, cross_attn_Wq, cross_attn.Q)
      .SET_MATRIX_LAYER(TextDecoder, cross_attn_Wk, cross_attn.K)
      .SET_MATRIX_LAYER(TextDecoder, cross_attn_Wv, cross_attn.V)
      .SET_MATRIX_LAYER(TextDecoder, cross_attn_Wout, cross_attn.out)
      .SET_VECTOR_LAYER(TextDecoder, cross_attn_out_bias, cross_attn.out_bias)
      .SET_VECTOR_LAYER(TextDecoder, cross_attn_q_bias, cross_attn.q_bias)
      .SET_VECTOR_LAYER(TextDecoder, cross_attn_v_bias, cross_attn.v_bias)
      .SET_VECTOR_LAYER(TextDecoder, mlp_ln_gamma, mlp_ln_gamma)
      .SET_VECTOR_LAYER(TextDecoder, mlp_ln_beta, mlp_ln_beta)
      .SET_VECTOR_LAYER(TextDecoder, attn_ln_gamma, attn_ln_gamma)
      .SET_VECTOR_LAYER(TextDecoder, attn_ln_beta, attn_ln_beta)
      .SET_VECTOR_LAYER(TextDecoder, cross_attn_ln_gamma, cross_attn_ln_gamma)
      .SET_VECTOR_LAYER(TextDecoder, cross_attn_ln_beta, cross_attn_ln_beta)
      .SET_VECTOR_DIRECT(TextDecoder, positional_embedding,
                         positional_embedding)
      .SET_VECTOR_DIRECT(TextDecoder, ln_gamma, ln_gamma)
      .SET_VECTOR_DIRECT(TextDecoder, ln_beta, ln_beta)
      .SET_MATRIX_DIRECT(TextDecoder, detokenizer0, detokenizer0)
      .SET_MATRIX_DIRECT(TextDecoder, detokenizer1, detokenizer1)
      .SET_MATRIX_DIRECT(TextDecoder, detokenizer2, detokenizer2)
      .def("reset",
           [](TextDecoder &dec, numpy_float_array x) {
             const float *data = static_cast<const float *>(x.request().ptr);
             auto ndim = x.ndim();
             auto shape = x.shape();
             assert((ndim == 2 || (ndim == 3 && shape[0] == 1)) &&
                    "Should be 2 dimensional or batch_size should be 1");
             int n_ctx = ndim == 2 ? shape[0] : shape[1];
             int n_state = ndim == 2 ? shape[1] : shape[2];
             int n_layer = dec.blocks.blocks.size();
             assert(n_ctx == dec.blocks.blocks[0]->cross_attn.kv_size &&
                    "n_ctx should be same as n_audio_ctx");
             for (int i = 0; i < n_layer; ++i) {
               dec.blocks.blocks[i]->cross_attn.K.set_A(data);
               dec.blocks.blocks[i]->reset();
             }
           })
      .def("call", [](TextDecoder &dec, int prompt) -> numpy_float_array {
        dec.call(prompt);

        int last_layer_idx = dec.blocks.blocks.size() - 1;
        int n_vocab = dec.n_vocab;
        int slice_len = n_vocab / 3;
        int last_slice_len = n_vocab - 2 * n_vocab / 3;
        auto y = numpy_float_array(1 * n_vocab);
        float *y_ptr = static_cast<float *>(y.request().ptr);
        for (int i = 0; i < slice_len; ++i) {
          y_ptr[i] = dec.detokenizer0.C_at(0, i);
          y_ptr[i + slice_len] = dec.detokenizer1.C_at(0, i);
	}
        for (int i = 0; i < last_slice_len; ++i) {
          y_ptr[i + 2 * slice_len] = dec.detokenizer2.C_at(0, i);
        }
        return y.reshape({1, n_vocab});
      });

  py::class_<WhisperModel>(m, "WhisperModel")
      .def(py::init<int, int, int, int, int, int, int, int, int, int>())
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, encoder_attn_Wq, attn.Q, encoder)
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, encoder_attn_Wk, attn.K, encoder)
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, encoder_attn_Wv, attn.V, encoder)
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, encoder_attn_Wout, attn.out,
                               encoder)
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, encoder_Wfc1, fc1, encoder)
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, encoder_Wfc2, fc2, encoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, encoder_attn_out_bias,
                               attn.out_bias, encoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, encoder_attn_q_bias, attn.q_bias,
                               encoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, encoder_attn_v_bias, attn.v_bias,
                               encoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, encoder_fc1_bias, fc1_bias,
                               encoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, encoder_fc2_bias, fc2_bias,
                               encoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, encoder_mlp_ln_gamma, mlp_ln_gamma,
                               encoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, encoder_mlp_ln_beta, mlp_ln_beta,
                               encoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, encoder_attn_ln_gamma,
                               attn_ln_gamma, encoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, encoder_attn_ln_beta, attn_ln_beta,
                               encoder)
      .SET_VECTOR_DIRECT(WhisperModel, encoder_positional_embedding,
                         encoder.positional_embedding)
      .SET_CONV_DIRECT(WhisperModel, conv0_weights, encoder.conv0)
      .SET_VECTOR_DIRECT(WhisperModel, conv0_bias, encoder.conv0_bias)
      .SET_CONV_DIRECT(WhisperModel, conv1_weights, encoder.conv1)
      .SET_VECTOR_DIRECT(WhisperModel, conv1_bias, encoder.conv1_bias)
      .SET_VECTOR_DIRECT(WhisperModel, encoder_ln_post_gamma,
                         encoder.ln_post_gamma)
      .SET_VECTOR_DIRECT(WhisperModel, encoder_ln_post_beta,
                         encoder.ln_post_beta)
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, decoder_attn_Wq, attn.Q, decoder)
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, decoder_attn_Wk, attn.K, decoder)
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, decoder_attn_Wv, attn.V, decoder)
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, decoder_attn_Wout, attn.out,
                               decoder)
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, decoder_Wfc1, fc1, decoder)
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, decoder_Wfc2, fc2, decoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, decoder_attn_out_bias,
                               attn.out_bias, decoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, decoder_attn_q_bias, attn.q_bias,
                               decoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, decoder_attn_v_bias, attn.v_bias,
                               decoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, decoder_fc1_bias, fc1_bias,
                               decoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, decoder_fc2_bias, fc2_bias,
                               decoder)
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, decoder_cross_attn_Wq,
                               cross_attn.Q, decoder)
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, decoder_cross_attn_Wk,
                               cross_attn.K, decoder)
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, decoder_cross_attn_Wv,
                               cross_attn.V, decoder)
      .SET_MATRIX_LAYER_PREFIX(WhisperModel, decoder_cross_attn_Wout,
                               cross_attn.out, decoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, decoder_cross_attn_out_bias,
                               cross_attn.out_bias, decoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, decoder_cross_attn_q_bias,
                               cross_attn.q_bias, decoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, decoder_cross_attn_v_bias,
                               cross_attn.v_bias, decoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, decoder_mlp_ln_gamma, mlp_ln_gamma,
                               decoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, decoder_mlp_ln_beta, mlp_ln_beta,
                               decoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, decoder_attn_ln_gamma,
                               attn_ln_gamma, decoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, decoder_attn_ln_beta, attn_ln_beta,
                               decoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, decoder_cross_attn_ln_gamma,
                               cross_attn_ln_gamma, decoder)
      .SET_VECTOR_LAYER_PREFIX(WhisperModel, decoder_cross_attn_ln_beta,
                               cross_attn_ln_beta, decoder)
      .SET_VECTOR_DIRECT(WhisperModel, decoder_positional_embedding,
                         decoder.positional_embedding)
      .SET_VECTOR_DIRECT(WhisperModel, decoder_ln_gamma, decoder.ln_gamma)
      .SET_VECTOR_DIRECT(WhisperModel, decoder_ln_beta, decoder.ln_beta)
      .SET_MATRIX_DIRECT(WhisperModel, detokenizer0, decoder.detokenizer0)
      .SET_MATRIX_DIRECT(WhisperModel, detokenizer1, decoder.detokenizer1)
      .SET_MATRIX_DIRECT(WhisperModel, detokenizer2, decoder.detokenizer2)
      .def("reset",
           [](WhisperModel &model, numpy_float_array x) {
             AudioEncoder *encoder = &model.encoder;
             TextDecoder *decoder = &model.decoder;
             const float *data = static_cast<const float *>(x.request().ptr);
             auto ndim = x.ndim();
             auto shape = x.shape();
             assert((ndim == 2 || (ndim == 3 && shape[0] == 1)) &&
                    "Should be 2 dimensional or batch_size should be 1");
             int n_ctx = ndim == 2 ? shape[0] : shape[1];
             int n_state = ndim == 2 ? shape[1] : shape[2];
             int n_text_layer = model.n_text_layer;
             assert(n_ctx / 2 == model.n_audio_ctx &&
                    "n_ctx should be same as n_audio_ctx");
             encoder->conv0.copy_A(data);
             encoder->call();
             int last_layer_idx = model.n_audio_layer - 1;
             const __fp16 *y_ptr =
                 encoder->blocks.blocks[last_layer_idx]->y.data();
             for (int i = 0; i < n_text_layer; ++i) {
               decoder->blocks.blocks[i]->cross_attn.K.set_A(y_ptr);
               decoder->blocks.blocks[i]->reset();
             }
           })
      .def("call",
           [](WhisperModel &model, int prompt) -> numpy_float_array {
             TextDecoder *decoder = &model.decoder;
             decoder->call(prompt);

             int last_layer_idx = model.n_text_layer - 1;
             int n_vocab = model.n_vocab;
             int slice_len = n_vocab / 3;
	     int last_slice_len = n_vocab - 2 * n_vocab / 3;
             auto y = numpy_float_array(1 * n_vocab);
             float *y_ptr = static_cast<float *>(y.request().ptr);
             for (int i = 0; i < slice_len; ++i) {
               y_ptr[i] = decoder->detokenizer0.C_at(0, i);
               y_ptr[i + slice_len] = decoder->detokenizer1.C_at(0, i);
	     }
             for (int i = 0; i < last_slice_len; ++i) {
               y_ptr[i + 2 * slice_len] = decoder->detokenizer2.C_at(0, i);
             }
             return y.reshape({1, n_vocab});
           })
      .def("call_no_copy",
           [](WhisperModel &model, int prompt) {
             TextDecoder *decoder = &model.decoder;
             decoder->call(prompt);
           })
      .def("get_logits",
           [](WhisperModel &model) {
             int n_vocab = model.n_vocab;
             TextDecoder *decoder = &model.decoder;
             auto y = numpy_uint16_array(n_vocab);
             __fp16 *y_ptr = static_cast<__fp16 *>(y.request().ptr);
             decoder->get_logits(y_ptr);
             return y.reshape({n_vocab});
           })
      .def("log_softmax",
           [](WhisperModel &model,
              const std::vector<int> &suppress_tokens) -> numpy_uint16_array {
             int n_vocab = model.n_vocab;
             TextDecoder *decoder = &model.decoder;

             auto y = numpy_uint16_array(n_vocab);
             __fp16 *y_ptr = static_cast<__fp16 *>(y.request().ptr);

             decoder->log_softmax(y_ptr, suppress_tokens);
             return y.reshape({n_vocab});
           });
}
