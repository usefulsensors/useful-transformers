#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>

#include "residual_attention_block.h"

namespace py = pybind11;

#define XSTR(s) STR(s)
#define STR(s) #s

#define SET_MATRIX(ClassName, m, M)                                      \
  def(STR(set_##m),                                                      \
      [](ClassName &obj, py::array_t<float, py::array::c_style> W) {     \
        const float *data = static_cast<const float *>(W.request().ptr); \
        obj.M.set_B(data);                                               \
      })

#define SET_VECTOR(ClassName, v, V)                                        \
  def(STR(set_##v),                                                        \
      [](ClassName &obj, py::array_t<float, py::array::c_style> vec) {     \
        const float *data = static_cast<const float *>(vec.request().ptr); \
        for (int i = 0; i < obj.V.size(); ++i) obj.V[i] = data[i];         \
      })

#define SET_MATRIX_LAYER(ClassName, m, M)                                   \
  def(STR(set_##m),                                                         \
      [](ClassName &obj, int i, py::array_t<float, py::array::c_style> W) { \
        const float *data = static_cast<const float *>(W.request().ptr);    \
        obj.blocks[i]->M.set_B(data);                                       \
      })

#define SET_VECTOR_LAYER(ClassName, v, V)                                     \
  def(STR(set_##v),                                                           \
      [](ClassName &obj, int i, py::array_t<float, py::array::c_style> vec) { \
        const float *data = static_cast<const float *>(vec.request().ptr);    \
        int sz = obj.blocks[i]->V.size();                                     \
        for (int j = 0; j < sz; ++j) obj.blocks[i]->V[j] = data[j];           \
      })

PYBIND11_MODULE(pybind_residual_attention_block, m) {
  m.doc() = "Residual Attention Blocks";
  using numpy_float_array = py::array_t<float, py::array::c_style>;
  py::class_<ResidualCrossAttentionBlock>(m, "ResidualCrossAttentionBlock")
      .def(py::init<int, int, int, int>())
      .SET_VECTOR(ResidualCrossAttentionBlock, attn_ln_gamma, attn_ln_gamma)
      .SET_VECTOR(ResidualCrossAttentionBlock, attn_ln_beta, attn_ln_beta)
      .SET_MATRIX(ResidualCrossAttentionBlock, attn_Wq, attn.Q)
      .SET_VECTOR(ResidualCrossAttentionBlock, attn_q_bias, attn.q_bias)
      .SET_MATRIX(ResidualCrossAttentionBlock, attn_Wk, attn.K)
      .SET_MATRIX(ResidualCrossAttentionBlock, attn_Wv, attn.V)
      .SET_VECTOR(ResidualCrossAttentionBlock, attn_v_bias, attn.v_bias)
      .SET_MATRIX(ResidualCrossAttentionBlock, attn_Wout, attn.out)
      .SET_VECTOR(ResidualCrossAttentionBlock, attn_out_bias, attn.out_bias)
      .SET_VECTOR(ResidualCrossAttentionBlock, cross_attn_ln_gamma,
                  cross_attn_ln_gamma)
      .SET_VECTOR(ResidualCrossAttentionBlock, cross_attn_ln_beta,
                  cross_attn_ln_beta)
      .SET_MATRIX(ResidualCrossAttentionBlock, cross_attn_Wq, cross_attn.Q)
      .SET_VECTOR(ResidualCrossAttentionBlock, cross_attn_q_bias,
                  cross_attn.q_bias)
      .SET_MATRIX(ResidualCrossAttentionBlock, cross_attn_Wk, cross_attn.K)
      .SET_MATRIX(ResidualCrossAttentionBlock, cross_attn_Wv, cross_attn.V)
      .SET_VECTOR(ResidualCrossAttentionBlock, cross_attn_v_bias,
                  cross_attn.v_bias)
      .SET_MATRIX(ResidualCrossAttentionBlock, cross_attn_Wout, cross_attn.out)
      .SET_VECTOR(ResidualCrossAttentionBlock, cross_attn_out_bias,
                  cross_attn.out_bias)
      .SET_VECTOR(ResidualCrossAttentionBlock, mlp_ln_gamma, mlp_ln_gamma)
      .SET_VECTOR(ResidualCrossAttentionBlock, mlp_ln_beta, mlp_ln_beta)
      .SET_MATRIX(ResidualCrossAttentionBlock, Wfc1, fc1)
      .SET_VECTOR(ResidualCrossAttentionBlock, fc1_bias, fc1_bias)
      .SET_MATRIX(ResidualCrossAttentionBlock, Wfc2, fc2)
      .SET_VECTOR(ResidualCrossAttentionBlock, fc2_bias, fc2_bias)
      .def("reset",
           [](ResidualCrossAttentionBlock &obj, numpy_float_array x) {
             const float *data = static_cast<const float *>(x.request().ptr);
             auto ndim = x.ndim();
             auto shape = x.shape();
             assert((ndim == 2 || (ndim == 3 && shape[0] == 1)) &&
                    "Should be 2 dimensional or batch_size should be 1");
             int n_ctx = ndim == 2 ? shape[0] : shape[1];
             int n_state = ndim == 2 ? shape[1] : shape[2];
             assert(n_ctx == obj.cross_attn.kv_size &&
                    "n_ctx should be same as n_audio_ctx");
             obj.cross_attn.K.set_A(data);
             obj.reset();
           })
      .def("call",
           [](ResidualCrossAttentionBlock &obj,
              numpy_float_array x) -> numpy_float_array {
             auto ndim = x.ndim();
             auto shape = x.shape();
             assert((ndim == 2 || (ndim == 3 && shape[0] == 1)) &&
                    "Should be 2 dimensional or batch_size should be 1");
             int n_ctx = ndim == 2 ? shape[0] : shape[1];
             int n_state = ndim == 2 ? shape[1] : shape[2];
             assert(n_state == obj.n_state && "n_state should match");
             const float *data = static_cast<const float *>(x.request().ptr);
             obj.attn.Q.set_A(data, n_ctx);
             obj.call(n_ctx);
             auto y = numpy_float_array(n_ctx * n_state);
             float *y_ptr = static_cast<float *>(y.request().ptr);
             for (int i = 0; i < n_ctx; ++i) {
               for (int j = 0; j < n_state; ++j) {
                 y_ptr[i * n_state + j] = obj.y[i * n_state + j];
               }
             }
             return y.reshape({n_ctx, n_state});
           });

  py::class_<ResidualCrossAttentionLayers>(m, "ResidualCrossAttentionLayers")
      .def(py::init<int, int, int, int, int>())
      .SET_VECTOR_LAYER(ResidualCrossAttentionLayers, attn_ln_gamma,
                        attn_ln_gamma)
      .SET_VECTOR_LAYER(ResidualCrossAttentionLayers, attn_ln_beta,
                        attn_ln_beta)
      .SET_MATRIX_LAYER(ResidualCrossAttentionLayers, attn_Wq, attn.Q)
      .SET_VECTOR_LAYER(ResidualCrossAttentionLayers, attn_q_bias, attn.q_bias)
      .SET_MATRIX_LAYER(ResidualCrossAttentionLayers, attn_Wk, attn.K)
      .SET_MATRIX_LAYER(ResidualCrossAttentionLayers, attn_Wv, attn.V)
      .SET_VECTOR_LAYER(ResidualCrossAttentionLayers, attn_v_bias, attn.v_bias)
      .SET_MATRIX_LAYER(ResidualCrossAttentionLayers, attn_Wout, attn.out)
      .SET_VECTOR_LAYER(ResidualCrossAttentionLayers, attn_out_bias,
                        attn.out_bias)
      .SET_VECTOR_LAYER(ResidualCrossAttentionLayers, cross_attn_ln_gamma,
                        cross_attn_ln_gamma)
      .SET_VECTOR_LAYER(ResidualCrossAttentionLayers, cross_attn_ln_beta,
                        cross_attn_ln_beta)
      .SET_MATRIX_LAYER(ResidualCrossAttentionLayers, cross_attn_Wq,
                        cross_attn.Q)
      .SET_VECTOR_LAYER(ResidualCrossAttentionLayers, cross_attn_q_bias,
                        cross_attn.q_bias)
      .SET_MATRIX_LAYER(ResidualCrossAttentionLayers, cross_attn_Wk,
                        cross_attn.K)
      .SET_MATRIX_LAYER(ResidualCrossAttentionLayers, cross_attn_Wv,
                        cross_attn.V)
      .SET_VECTOR_LAYER(ResidualCrossAttentionLayers, cross_attn_v_bias,
                        cross_attn.v_bias)
      .SET_MATRIX_LAYER(ResidualCrossAttentionLayers, cross_attn_Wout,
                        cross_attn.out)
      .SET_VECTOR_LAYER(ResidualCrossAttentionLayers, cross_attn_out_bias,
                        cross_attn.out_bias)
      .SET_VECTOR_LAYER(ResidualCrossAttentionLayers, mlp_ln_gamma,
                        mlp_ln_gamma)
      .SET_VECTOR_LAYER(ResidualCrossAttentionLayers, mlp_ln_beta, mlp_ln_beta)
      .SET_MATRIX_LAYER(ResidualCrossAttentionLayers, Wfc1, fc1)
      .SET_VECTOR_LAYER(ResidualCrossAttentionLayers, fc1_bias, fc1_bias)
      .SET_MATRIX_LAYER(ResidualCrossAttentionLayers, Wfc2, fc2)
      .SET_VECTOR_LAYER(ResidualCrossAttentionLayers, fc2_bias, fc2_bias)
      .def("reset",
           [](ResidualCrossAttentionLayers &obj, numpy_float_array x) {
             const float *data = static_cast<const float *>(x.request().ptr);
             auto ndim = x.ndim();
             auto shape = x.shape();
             assert((ndim == 2 || (ndim == 3 && shape[0] == 1)) &&
                    "Should be 2 dimensional or batch_size should be 1");
             int n_ctx = ndim == 2 ? shape[0] : shape[1];
             int n_state = ndim == 2 ? shape[1] : shape[2];
             int n_layer = obj.blocks.size();
             assert(n_ctx == obj.blocks[0]->cross_attn.kv_size &&
                    "n_ctx should be same as n_audio_ctx");
             for (int i = 0; i < n_layer; ++i) {
               obj.blocks[i]->cross_attn.K.set_A(data);
               obj.blocks[i]->reset();
             }
           })
      .def(
          "call",
          [](ResidualCrossAttentionLayers &obj,
             numpy_float_array x) -> numpy_float_array {
            auto ndim = x.ndim();
            auto shape = x.shape();
            assert((ndim == 2 || (ndim == 3 && shape[0] == 1)) &&
                   "Should be 2 dimensional or batch_size should be 1");
            int n_ctx = ndim == 2 ? shape[0] : shape[1];
            int n_state = ndim == 2 ? shape[1] : shape[2];
            assert(n_state == obj.blocks[0]->n_state && "n_state should match");
            const float *data = static_cast<const float *>(x.request().ptr);
            obj.blocks[0]->attn.Q.set_A(data, n_ctx);
            obj.call(n_ctx);
            auto y = numpy_float_array(n_ctx * n_state);
            float *y_ptr = static_cast<float *>(y.request().ptr);
            int last_block_idx = obj.blocks.size() - 1;
            for (int i = 0; i < n_ctx; ++i) {
              for (int j = 0; j < n_state; ++j) {
                y_ptr[i * n_state + j] =
                    obj.blocks[last_block_idx]->y[i * n_state + j];
              }
            }
            return y.reshape({n_ctx, n_state});
          });

  py::class_<ResidualSelfAttentionBlock>(m, "ResidualSelfAttentionBlock")
      .def(py::init<int, int, int>())
      .SET_VECTOR(ResidualSelfAttentionBlock, attn_ln_gamma, attn_ln_gamma)
      .SET_VECTOR(ResidualSelfAttentionBlock, attn_ln_beta, attn_ln_beta)
      .SET_MATRIX(ResidualSelfAttentionBlock, attn_Wq, attn.Q)
      .SET_VECTOR(ResidualSelfAttentionBlock, attn_q_bias, attn.q_bias)
      .SET_MATRIX(ResidualSelfAttentionBlock, attn_Wk, attn.K)
      .SET_MATRIX(ResidualSelfAttentionBlock, attn_Wv, attn.V)
      .SET_VECTOR(ResidualSelfAttentionBlock, attn_v_bias, attn.v_bias)
      .SET_MATRIX(ResidualSelfAttentionBlock, attn_Wout, attn.out)
      .SET_VECTOR(ResidualSelfAttentionBlock, attn_out_bias, attn.out_bias)
      .SET_VECTOR(ResidualSelfAttentionBlock, mlp_ln_gamma, mlp_ln_gamma)
      .SET_VECTOR(ResidualSelfAttentionBlock, mlp_ln_beta, mlp_ln_beta)
      .SET_MATRIX(ResidualSelfAttentionBlock, Wfc1, fc1)
      .SET_VECTOR(ResidualSelfAttentionBlock, fc1_bias, fc1_bias)
      .SET_MATRIX(ResidualSelfAttentionBlock, Wfc2, fc2)
      .SET_VECTOR(ResidualSelfAttentionBlock, fc2_bias, fc2_bias)
      .def("call",
           [](ResidualSelfAttentionBlock &obj,
              numpy_float_array x) -> numpy_float_array {
             auto ndim = x.ndim();
             auto shape = x.shape();
             assert((ndim == 2 || (ndim == 3 && shape[0] == 1)) &&
                    "Should be 2 dimensional or batch_size should be 1");
             int n_ctx = ndim == 2 ? shape[0] : shape[1];
             int n_state = ndim == 2 ? shape[1] : shape[2];
             assert(n_ctx == obj.n_ctx && "n_ctx should match");
             assert(n_state == obj.n_state && "n_state should match");
             const float *data = static_cast<const float *>(x.request().ptr);
             obj.attn.Q.set_A(data, n_ctx);
             obj.call();
             auto y = numpy_float_array(n_ctx * n_state);
             float *y_ptr = static_cast<float *>(y.request().ptr);
             for (int i = 0; i < n_ctx; ++i) {
               for (int j = 0; j < n_state; ++j) {
                 y_ptr[i * n_state + j] = obj.y[i * n_state + j];
               }
             }
             return y.reshape({n_ctx, n_state});
           });

  py::class_<ResidualSelfAttentionLayers>(m, "ResidualSelfAttentionLayers")
      .def(py::init<int, int, int, int>())
      .SET_VECTOR_LAYER(ResidualSelfAttentionLayers, attn_ln_gamma,
                        attn_ln_gamma)
      .SET_VECTOR_LAYER(ResidualSelfAttentionLayers, attn_ln_beta, attn_ln_beta)
      .SET_MATRIX_LAYER(ResidualSelfAttentionLayers, attn_Wq, attn.Q)
      .SET_VECTOR_LAYER(ResidualSelfAttentionLayers, attn_q_bias, attn.q_bias)
      .SET_MATRIX_LAYER(ResidualSelfAttentionLayers, attn_Wk, attn.K)
      .SET_MATRIX_LAYER(ResidualSelfAttentionLayers, attn_Wv, attn.V)
      .SET_VECTOR_LAYER(ResidualSelfAttentionLayers, attn_v_bias, attn.v_bias)
      .SET_MATRIX_LAYER(ResidualSelfAttentionLayers, attn_Wout, attn.out)
      .SET_VECTOR_LAYER(ResidualSelfAttentionLayers, attn_out_bias,
                        attn.out_bias)
      .SET_VECTOR_LAYER(ResidualSelfAttentionLayers, mlp_ln_gamma, mlp_ln_gamma)
      .SET_VECTOR_LAYER(ResidualSelfAttentionLayers, mlp_ln_beta, mlp_ln_beta)
      .SET_MATRIX_LAYER(ResidualSelfAttentionLayers, Wfc1, fc1)
      .SET_VECTOR_LAYER(ResidualSelfAttentionLayers, fc1_bias, fc1_bias)
      .SET_MATRIX_LAYER(ResidualSelfAttentionLayers, Wfc2, fc2)
      .SET_VECTOR_LAYER(ResidualSelfAttentionLayers, fc2_bias, fc2_bias)
      .def(
          "call",
          [](ResidualSelfAttentionLayers &obj,
             numpy_float_array x) -> numpy_float_array {
            auto ndim = x.ndim();
            auto shape = x.shape();
            assert((ndim == 2 || (ndim == 3 && shape[0] == 1)) &&
                   "Should be 2 dimensional or batch_size should be 1");
            int n_ctx = ndim == 2 ? shape[0] : shape[1];
            int n_state = ndim == 2 ? shape[1] : shape[2];
            assert(n_ctx == obj.blocks[0]->n_ctx && "n_ctx should match");
            assert(n_state == obj.blocks[0]->n_state && "n_state should match");
            const float *data = static_cast<const float *>(x.request().ptr);
            obj.blocks[0]->attn.Q.set_A(data, n_ctx);
            obj.call();
            auto y = numpy_float_array(n_ctx * n_state);
            float *y_ptr = static_cast<float *>(y.request().ptr);
            int last_block_idx = obj.blocks.size() - 1;
            for (int i = 0; i < n_ctx; ++i) {
              for (int j = 0; j < n_state; ++j) {
                y_ptr[i * n_state + j] =
                    obj.blocks[last_block_idx]->y[i * n_state + j];
              }
            }
            return y.reshape({n_ctx, n_state});
          });
}
