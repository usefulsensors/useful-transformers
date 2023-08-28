#include "matmul.h"

#include <cassert>
#include <cstring>

Matmul::Matmul(int M, int K, int N, int core) : M(M), K(K), N(N) {
  K_padded = NEXT_MULTIPLE_OF_32(K);
  N_padded = NEXT_MULTIPLE_OF_16(N);
  memset(&info, 0, sizeof(rknn_matmul_info));
  info.M = M;
  info.K = K_padded;
  info.N = N_padded;
  info.type = RKNN_TENSOR_FLOAT16;
  info.native_layout = 1;
  info.perf_layout = 1;
  memset(&io_attr, 0, sizeof(rknn_matmul_io_attr));

  int ret = 0;

  ret = rknn_matmul_create(&ctx, &info, &io_attr);
  assert(ret >= 0 && "Constructor failed");

  if (core == 0)
    ret = rknn_matmul_set_core_mask(ctx, RKNN_NPU_CORE_0);
  else if (core == 1)
    ret = rknn_matmul_set_core_mask(ctx, RKNN_NPU_CORE_1);
  else if (core == 2)
    ret = rknn_matmul_set_core_mask(ctx, RKNN_NPU_CORE_2);
  else
    ret = -1;
  assert(ret >= 0 && "Set core mask failed");

  assert(io_attr.A.n_dims == 3 && "Expected A to be 3 dimensional");
  assert(io_attr.B.n_dims == 4 && "Expected B to be 4 dimensional");
  assert(io_attr.C.n_dims == 3 && "Expected C to be 3 dimensional");

  assert(io_attr.A.dims[2] == 8 && io_attr.A.dims[1] == M &&
         io_attr.A.dims[0] == K_padded / 8 && "A dims are not [K/8, M, 8]");
  assert(io_attr.B.dims[3] == 32 && io_attr.B.dims[2] == 16 &&
         io_attr.B.dims[1] == K_padded / 32 &&
         io_attr.B.dims[0] == N_padded / 16 &&
         "B dims are not [N/16, K/32, 16, 32]");
  assert(io_attr.C.dims[2] == 4 && io_attr.C.dims[1] == M &&
         io_attr.C.dims[0] == N_padded / 4 && "C dims are not [N/4, M, 4]");

  A = rknn_create_mem(ctx, io_attr.A.size);
  assert(A != NULL && "A allocation failed");
  B = rknn_create_mem(ctx, io_attr.B.size);
  assert(B != NULL && "B allocation failed");
  C = rknn_create_mem(ctx, io_attr.C.size);
  assert(C != NULL && "C allocation failed");

  zero_A();
  zero_B();
}

Matmul::~Matmul() {
  rknn_destroy_mem(ctx, A);
  rknn_destroy_mem(ctx, B);
  rknn_destroy_mem(ctx, C);

  rknn_matmul_destroy(ctx);
}

void Matmul::zero_A() { memset(get_A_ptr(), 0, sizeof(__fp16) * M * K_padded); }

void Matmul::zero_B() {
  memset(get_B_ptr(), 0, sizeof(__fp16) * K_padded * N_padded);
}

void Matmul::copy_B_to_B(Matmul* other) {
  other->zero_B();
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      other->B_at(i, j) = B_at(i, j);
    }
  }
}

void Matmul::copy_C_to_A(Matmul* other, Slice slice) {
  for (int i = 0; i < slice.n_rows; ++i) {
    for (int j = 0; j < slice.n_cols; ++j) {
      other->A_at(i + slice.dst_row_offset, j + slice.dst_col_offset) =
          C_at(i + slice.src_row_offset, j + slice.src_col_offset);
    }
  }
}

void Matmul::copy_C_to_Bt(Matmul* other, Slice slice) {
  for (int i = 0; i < slice.n_rows; ++i) {
    for (int j = 0; j < slice.n_cols; ++j) {
      other->B_at(j + slice.dst_col_offset, i + slice.dst_row_offset) =
          C_at(i + slice.src_row_offset, j + slice.src_col_offset);
    }
  }
}

void Matmul::copy_C_to_B(Matmul* other, Slice slice) {
  for (int i = 0; i < slice.n_rows; ++i) {
    for (int j = 0; j < slice.n_cols; ++j) {
      other->B_at(i + slice.dst_row_offset, j + slice.dst_col_offset) =
          C_at(i + slice.src_row_offset, j + slice.src_col_offset);
    }
  }
}

void Matmul::call() {
  int ret;
  ret = rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
  assert(ret >= 0 && "Setting A input failed");
  ret = rknn_matmul_set_io_mem(ctx, B, &io_attr.B);
  assert(ret >= 0 && "Setting B input failed");
  ret = rknn_matmul_set_io_mem(ctx, C, &io_attr.C);
  assert(ret >= 0 && "Setting C ouput failed");
  ret = rknn_matmul_run(ctx);
  assert(ret >= 0 && "matmul launch failed\n");
}
