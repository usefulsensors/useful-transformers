#ifndef _LIB_MATVIEW_H_
#define _LIB_MATVIEW_H_

#include <cassert>

template <typename T>
struct MatView {
  T *ptr;
  int S0, S1, S2, S3;

  MatView(T *ptr, int S2, int S3) : ptr(ptr), S0(1), S1(1), S2(S2), S3(S3) {
    check();
  }

  MatView(T *ptr, int S1, int S2, int S3)
      : ptr(ptr), S0(1), S1(S1), S2(S2), S3(S3) {
    check();
  }

  MatView(T *ptr, int S0, int S1, int S2, int S3)
      : ptr(ptr), S0(S0), S1(S1), S2(S2), S3(S3) {
    check();
  }

  void check() const {
    assert(0 < S0 && 0 < S1 && 0 < S2 && 0 < S3 && "Sizes should be > 0");
  }

  void check(int i, int j, int k, int l) const {
    assert(0 <= i && i < S0 && "i index out of range");
    assert(0 <= j && j < S1 && "j index out of range");
    assert(0 <= k && k < S2 && "k index out of range");
    assert(0 <= l && l < S3 && "l index out of range");
  }

  T &operator()(int k, int l) {
    return this->operator()(0, 0, k, l);
  }

  const T &operator()(int k, int l) const {
    return this->operator()(0, 0, k, l);
  }

  T &operator()(int j, int k, int l) { return this->operator()(0, j, k, l); }

  const T &operator()(int j, int k, int l) const {
    return this->operator()(0, j, k, l);
  }

  T &operator()(int i, int j, int k, int l) {
    check(i, j, k, l);
    return ptr[((i * S1 + j) * S2 + k) * S3 + l];
  }

  const T &operator()(int i, int j, int k, int l) const {
    check(i, j, k, l);
    return ptr[((i * S1 + j) * S2 + k) * S3 + l];
  }
};
#endif  // _LIB_MATVIEW_H_
