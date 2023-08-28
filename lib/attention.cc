#include "attention.h"

#include <cassert>
#include <cmath>
#include <limits>

#include "bias.h"
#include "matmul.h"
#include "softmax.h"

CachingAttention::CachingAttention(int n_max_ctx, int n_state, int n_head)
    : n_max_ctx(n_max_ctx),
      n_state(n_state),
      n_head(n_head),
      inner_size(n_state / n_head),
      cur_kv_len(0),
      cur_kv_size(32),
      Q(n_max_ctx, n_state, n_state, 0),
      K(n_max_ctx, n_state, n_state, 1),
      V(n_max_ctx, n_state, n_state, 2),
      q_bias(n_state),
      v_bias(n_state),
      outer_product(n_head),
      w(cur_kv_size * n_max_ctx),
      wv(n_head),
      out(n_max_ctx, n_state, n_state),
      out_bias(n_state) {
  for (int i = 0; i < n_head; ++i) {
    outer_product[i] = new Matmul(n_max_ctx, inner_size, cur_kv_size, i % 3);
    wv[i] = new Matmul(n_max_ctx, cur_kv_size, inner_size, i % 3);
  }
}

CachingAttention::~CachingAttention() {
  for (int i = 0; i < n_head; ++i) {
    delete outer_product[i];
    delete wv[i];
  }
}

void CachingAttention::call(int n_ctx) {
  assert(n_ctx <= n_max_ctx && "n_ctx > n_max_ctx");

#pragma omp parallel sections
  {
#pragma omp section
    { Q.call(); }
#pragma omp section
    {
      // Assume x has been copied to Q.A
      memcpy(K.get_A_ptr(), Q.get_A_ptr(),
             sizeof(__fp16) * n_max_ctx * n_state);
      K.call();
    }
#pragma omp section
    {
      // Assume x has been copied to Q.A
      memcpy(V.get_A_ptr(), Q.get_A_ptr(),
             sizeof(__fp16) * n_max_ctx * n_state);
      V.call();
    }
  }

  float scale = std::pow((float)n_state / float(n_head), -0.25);

  for (int i = 0; i < n_ctx; ++i) {
    for (int j = 0; j < n_state; ++j) {
      float q_elem = Q.C_at(i, j);
      float k_elem = K.C_at(i, j);
      float v_elem = V.C_at(i, j);
      Q.C_at(i, j) = (q_elem + q_bias[j]) * scale;
      K.C_at(i, j) = k_elem * scale;
      V.C_at(i, j) = v_elem + v_bias[j];
    }
  }

  if (n_ctx + cur_kv_len > cur_kv_size) {
    int new_size =
        NEXT_MULTIPLE_OF_32(std::max(n_ctx + cur_kv_len, cur_kv_size * 2));

    w.resize(new_size * n_max_ctx);

    for (int i = 0; i < n_head; ++i) {
      Matmul* tmp = new Matmul(n_max_ctx, inner_size, new_size, i % 3);
      outer_product[i]->copy_B_to_B(tmp);
      delete outer_product[i];
      outer_product[i] = tmp;

      tmp = new Matmul(n_max_ctx, new_size, inner_size, i % 3);
      wv[i]->copy_B_to_B(tmp);
      delete wv[i];
      wv[i] = tmp;
    }
    cur_kv_size = new_size;
  }

  auto attn = [&](int ctx) {
    Q.copy_C_to_A(outer_product[ctx],
                  Matmul::Slice{.n_rows = n_ctx,
                                .n_cols = inner_size,
                                .src_col_offset = ctx * inner_size});

    K.copy_C_to_Bt(outer_product[ctx],
                   Matmul::Slice{.n_rows = n_ctx,
                                 .n_cols = inner_size,
                                 .dst_row_offset = cur_kv_len,
                                 .src_col_offset = ctx * inner_size});

    outer_product[ctx]->call();

    for (int i = 0; i < n_ctx; ++i) {
      for (int j = 0; j < n_ctx + cur_kv_len; ++j) {
        w[i * n_max_ctx + j] = outer_product[ctx]->C_at(i, j);
      }
    }

    /* Softmax */
    for (int i = 0; i < n_ctx; ++i) {
      float max = w[0 * n_max_ctx + i];
      for (int j = 1; j < n_ctx + cur_kv_len; ++j) {
        max = std::max(max, w[j * n_max_ctx + i]);
      }
      float sum = 0.0;
      for (int j = 0; j < n_ctx + cur_kv_len; ++j) {
        float elem = w[j * n_max_ctx + i] - max;
        elem = std::exp(elem);
        sum += elem;
        wv[ctx]->A_at(i, j) = elem;
      }
      for (int j = 0; j < n_ctx + cur_kv_len; ++j) {
        wv[ctx]->A_at(i, j) /= sum;
      }
    }

    /* Copy v to wv's B */
    V.copy_C_to_B(wv[ctx], Matmul::Slice{.n_rows = n_ctx,
                                         .n_cols = inner_size,
                                         .dst_row_offset = cur_kv_len,
                                         .src_col_offset = ctx * inner_size});

    wv[ctx]->call();

    wv[ctx]->copy_C_to_A(&out,
                         Matmul::Slice{.n_rows = n_ctx,
                                       .n_cols = inner_size,
                                       .dst_col_offset = ctx * inner_size});
  };

  for (int head = 0; head < n_head; head += 3) {
#pragma omp parallel sections
    {
#pragma omp section
      {
        if (head < n_head) attn(head);
      }
#pragma omp section
      {
        if (head + 1 < n_head) attn(head + 1);
      }
#pragma omp section
      {
        if (head + 2 < n_head) attn(head + 2);
      }
    }
  }

  cur_kv_len += n_ctx;

  out.call();

  for (int i = 0; i < n_ctx; ++i) {
    for (int j = 0; j < n_state; ++j) {
      out.C_at(i, j) += out_bias[j];
    }
  }
}

void CachingAttention::reset() {
  cur_kv_len = 0;
  int new_size = 32;

  w.resize(new_size * n_max_ctx);

  for (int i = 0; i < n_head; ++i) {
    delete outer_product[i];
    outer_product[i] = new Matmul(n_max_ctx, inner_size, new_size, i % 3);
    delete wv[i];
    wv[i] = new Matmul(n_max_ctx, new_size, inner_size, i % 3);
  }
  cur_kv_size = new_size;
}

ConstantKVAttention::ConstantKVAttention(int n_max_ctx, int n_state, int n_head,
                                         int kv_size)
    : n_max_ctx(n_max_ctx),
      n_state(n_state),
      n_head(n_head),
      inner_size(n_state / n_head),
      kv_size(kv_size),
      Q(n_max_ctx, n_state, n_state, 0),
      K(kv_size, n_state, n_state, 1),
      V(kv_size, n_state, n_state, 2),
      q_bias(n_state),
      v_bias(n_state),
      outer_product(n_head),
      wv(n_head),
      out(n_max_ctx, n_state, n_state),
      out_bias(n_state) {
  for (int i = 0; i < n_head; ++i) {
    outer_product[i] = new Matmul(n_max_ctx, inner_size, kv_size, i % 3);
    wv[i] = new Matmul(n_max_ctx, kv_size, inner_size, i % 3);
  }
}

ConstantKVAttention::~ConstantKVAttention() {
  for (int i = 0; i < n_head; ++i) {
    delete outer_product[i];
    delete wv[i];
  }
}

void ConstantKVAttention::call(int n_ctx) {
  assert(n_ctx <= n_max_ctx && "n_ctx > n_max_ctx");

  Q.call();

  float scale = std::pow((float)n_state / float(n_head), -0.25);

  for (int i = 0; i < n_ctx; ++i) {
    for (int j = 0; j < n_state; ++j) {
      float q_elem = Q.C_at(i, j);
      Q.C_at(i, j) = (q_elem + q_bias[j]) * scale;
    }
  }

  auto attn = [&](int ctx) {
    Q.copy_C_to_A(outer_product[ctx],
                  Matmul::Slice{.n_rows = n_ctx,
                                .n_cols = inner_size,
                                .src_col_offset = ctx * inner_size});

    outer_product[ctx]->call();

#ifdef NEON_OPT
    softmax_C_to_A(outer_product[ctx], wv[ctx], n_ctx, kv_size);
#else
    /* Softmax */
    for (int i = 0; i < n_ctx; ++i) {
      float max = outer_product[ctx]->C_at(i, 0);
      for (int j = 1; j < kv_size; ++j) {
        max = std::max(max, outer_product[ctx]->C_at(i, j));
      }
      float sum = 0.0;
      for (int j = 0; j < kv_size; ++j) {
        float elem = outer_product[ctx]->C_at(i, j) - max;
        elem = std::exp(elem);
        sum += elem;
        wv[ctx]->A_at(i, j) = elem;
      }
      for (int j = 0; j < kv_size; ++j) {
        wv[ctx]->A_at(i, j) /= sum;
      }
    }
#endif

    wv[ctx]->call();

    wv[ctx]->copy_C_to_A(&out,
                         Matmul::Slice{.n_rows = n_ctx,
                                       .n_cols = inner_size,
                                       .dst_col_offset = ctx * inner_size});
  };
  for (int head = 0; head < n_head; head += 3) {
#pragma omp parallel sections
    {
#pragma omp section
      {
        if (head < n_head) attn(head);
      }
#pragma omp section
      {
        if (head + 1 < n_head) attn(head + 1);
      }
#pragma omp section
      {
        if (head + 2 < n_head) attn(head + 2);
      }
    }
  }

  out.call();

  for (int i = 0; i < n_ctx; ++i) {
    for (int j = 0; j < n_state; ++j) {
      out.C_at(i, j) += out_bias[j];
    }
  }
}

void ConstantKVAttention::reset() {
  memcpy(V.get_A_ptr(), K.get_A_ptr(), sizeof(__fp16) * kv_size * n_state);

#pragma omp parallel sections
  {
#pragma omp section
    { K.call(); }
#pragma omp section
    { V.call(); }
  }

  float scale = std::pow((float)n_state / float(n_head), -0.25);

#ifdef NEON_OPT
  bias_and_scale_C(&K, std::vector<__fp16>(n_state, 0.0), scale, kv_size,
                   n_state);
  bias_and_scale_C(&V, v_bias, 1.0, kv_size, n_state);
#else
  for (int i = 0; i < kv_size; ++i) {
    for (int j = 0; j < n_state; ++j) {
      float k_elem = K.C_at(i, j);
      float v_elem = V.C_at(i, j);
      K.C_at(i, j) = k_elem * scale;
      V.C_at(i, j) = v_elem + v_bias[j];
    }
  }
#endif

  for (int i = 0; i < n_head; ++i) {
    K.copy_C_to_Bt(outer_product[i],
                   Matmul::Slice{.n_rows = kv_size,
                                 .n_cols = inner_size,
                                 .src_col_offset = i * inner_size});
    V.copy_C_to_B(wv[i], Matmul::Slice{.n_rows = kv_size,
                                       .n_cols = inner_size,
                                       .src_col_offset = i * inner_size});
  }
}

FixedShapeAttention::FixedShapeAttention(int n_ctx, int n_state, int n_head)
    : n_ctx(n_ctx),
      n_state(n_state),
      n_head(n_head),
      inner_size(n_state / n_head),
      Q(n_ctx, n_state, n_state, 0),
      K(n_ctx, n_state, n_state, 1),
      V(n_ctx, n_state, n_state, 2),
      q_bias(n_state),
      v_bias(n_state),
      outer_product(n_head),
      wv(n_head),
      out(n_ctx, n_state, n_state),
      out_bias(n_state) {
  for (int i = 0; i < n_head; ++i) {
    if (i < 3) {
      outer_product[i] = new Matmul(n_ctx, inner_size, n_ctx, i % 3);
      wv[i] = new Matmul(n_ctx, n_ctx, inner_size, i % 3);
    } else {
      outer_product[i] = outer_product[i % 3];
      wv[i] = wv[i % 3];
    }
  }
}

FixedShapeAttention::~FixedShapeAttention() {
  for (int i = 0; i < std::min(n_head, 3); ++i) {
    delete outer_product[i];
    delete wv[i];
  }
}

void FixedShapeAttention::call() {
  // Assume x has been copied to Q.A
  memcpy(K.get_A_ptr(), Q.get_A_ptr(), sizeof(__fp16) * n_ctx * n_state);
  memcpy(V.get_A_ptr(), Q.get_A_ptr(), sizeof(__fp16) * n_ctx * n_state);

#pragma omp parallel sections
  {
#pragma omp section
    { Q.call(); }
#pragma omp section
    { K.call(); }
#pragma omp section
    { V.call(); }
  }

  float scale = std::pow((float)n_state / float(n_head), -0.25);

#ifdef NEON_OPT
  bias_and_scale_C(&Q, q_bias, scale, n_ctx, n_state);
  bias_and_scale_C(&K, std::vector<__fp16>(n_state, 0.0), scale, n_ctx,
                   n_state);
  bias_and_scale_C(&V, v_bias, 1.0, n_ctx, n_state);
#else
  for (int i = 0; i < n_ctx; ++i) {
    for (int j = 0; j < n_state; ++j) {
      float q_elem = Q.C_at(i, j);
      float k_elem = K.C_at(i, j);
      float v_elem = V.C_at(i, j);
      Q.C_at(i, j) = (q_elem + q_bias[j]) * scale;
      K.C_at(i, j) = k_elem * scale;
      V.C_at(i, j) = v_elem + v_bias[j];
    }
  }
#endif

  auto attn = [&](int ctx) {
    Q.copy_C_to_A(outer_product[ctx],
                  Matmul::Slice{.n_rows = n_ctx,
                                .n_cols = inner_size,
                                .src_col_offset = ctx * inner_size});

    K.copy_C_to_Bt(outer_product[ctx],
                   Matmul::Slice{.n_rows = n_ctx,
                                 .n_cols = inner_size,
                                 .src_col_offset = ctx * inner_size});

    outer_product[ctx]->call();

#ifdef NEON_OPT
    softmax_C_to_A(outer_product[ctx], wv[ctx], n_ctx, n_ctx);
#else
    /* Softmax */
    for (int i = 0; i < n_ctx; ++i) {
      float max = outer_product[ctx]->C_at(i, 0);
      for (int j = 1; j < n_ctx; ++j) {
        max = std::max(max, outer_product[ctx]->C_at(i, j));
      }
      float sum = 0.0;
      for (int j = 0; j < n_ctx; ++j) {
        float elem = outer_product[ctx]->C_at(i, j) - max;
        elem = std::exp(elem);
        sum += elem;
        wv[ctx]->A_at(i, j) = elem;
      }
      for (int j = 0; j < n_ctx; ++j) {
        wv[ctx]->A_at(i, j) /= sum;
      }
    }
#endif

    /* Copy v to wv's B */
    V.copy_C_to_B(wv[ctx], Matmul::Slice{.n_rows = n_ctx,
                                         .n_cols = inner_size,
                                         .src_col_offset = ctx * inner_size});

    wv[ctx]->call();

    wv[ctx]->copy_C_to_A(&out,
                         Matmul::Slice{.n_rows = n_ctx,
                                       .n_cols = inner_size,
                                       .dst_col_offset = ctx * inner_size});
  };

  for (int head = 0; head < n_head; head += 3) {
#pragma omp parallel sections
    {
#pragma omp section
      {
        if (head < n_head) attn(head);
      }
#pragma omp section
      {
        if (head + 1 < n_head) attn(head + 1);
      }
#pragma omp section
      {
        if (head + 2 < n_head) attn(head + 2);
      }
    }
  }

  out.call();

#ifdef NEON_OPT
  bias_and_scale_C(&out, out_bias, 1.0, n_ctx, n_state);
#else
  for (int i = 0; i < n_ctx; ++i) {
    for (int j = 0; j < n_state; ++j) {
      out.C_at(i, j) += out_bias[j];
    }
  }
#endif
}
