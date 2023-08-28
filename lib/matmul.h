#ifndef _LIB_MATMUL_H_
#define _LIB_MATMUL_H_

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>

#include "rknn_matmul_api.h"

#define NEXT_MULTIPLE_OF_32(x) (((x) + 31) & ~31)
#define NEXT_MULTIPLE_OF_16(x) (((x) + 15) & ~15)

struct Matmul {
  int M = 0;
  int K = 0;
  int N = 0;

  int K_padded;
  int N_padded;

  rknn_tensor_mem *A = 0;
  rknn_tensor_mem *B = 0;
  rknn_tensor_mem *C = 0;

  rknn_matmul_ctx ctx;
  rknn_matmul_info info;
  rknn_matmul_io_attr io_attr;

  Matmul(int M, int K, int N, int core = 0);
  ~Matmul();

  __fp16 *get_A_ptr() { return (__fp16 *)((size_t)A->virt_addr + A->offset); }

  __fp16 *get_B_ptr() { return (__fp16 *)((size_t)B->virt_addr + B->offset); }

  float *get_C_ptr() { return (float *)((size_t)C->virt_addr + C->offset); }

  __fp16 &A_at(int i, int j) {
    assert(0 <= i && i < M && 0 <= j && j < K && "A indices out of range");
    // A [j/8, i, j%8] of [K_padded / 8, M, 8] array
    return get_A_ptr()[((j / 8) * M + i) * 8 + (j % 8)];
  }

  __fp16 &B_at(int i, int j) {
    assert(0 <= i && i < K && 0 <= j && j < N && "B indices out of range");
    // B [j/16, i/32, j%16, i%32] of [N_padded/16, K_padded/32, 16, 32] array
    return get_B_ptr()[((((j / 16) * (K_padded / 32)) + (i / 32)) * 16 +
                        (j % 16)) *
                           32 +
                       (i % 32)];
  }

  float &C_at(int i, int j) {
    assert(0 <= i && i < M && 0 <= j && j < N && "C indices out of range");
    // C [j/4, i, j%4] of [N_padded/4, M, 4] array
    return get_C_ptr()[(((j / 4) * M + i) * 4) + (j % 4)];
  }

  template <typename T>
  void set_A(const T *src) {
    zero_A();
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        A_at(i, j) = src[i * K + j];
      }
    }
  }

  template <typename T>
  void set_A(const T *src, int num_rows) {
    zero_A();
    for (int i = 0; i < num_rows; ++i) {
      for (int j = 0; j < K; ++j) {
        A_at(i, j) = src[i * K + j];
      }
    }
  }

  template <typename T>
  void set_B(const T *src) {
    zero_B();
    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < N; ++j) {
        B_at(i, j) = src[i * N + j];
      }
    }
  }

  template <typename T>
  void get_A(T *dst, int dst_rows, int dst_cols) {
    for (int i = 0; i < std::min(M, dst_rows); ++i) {
      for (int j = 0; j < std::min(K, dst_cols); ++j) {
        dst[i * dst_cols + j] = A_at(i, j);
      }
    }
  }

  void zero_A();
  void zero_B();
  void copy_B_to_B(Matmul *other);

  struct Slice {
    int n_rows;
    int n_cols;

    int src_row_offset = 0;
    int dst_row_offset = 0;

    int src_col_offset = 0;
    int dst_col_offset = 0;
  };

  void copy_C_to_A(Matmul *other, Slice slice);
  void copy_C_to_Bt(Matmul *other, Slice slice);
  void copy_C_to_B(Matmul *other, Slice slice);

  void call();
};

#endif  // _LIB_MATMUL_H_
