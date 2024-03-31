#pragma once

#include "utils/assert.cuh"
#include "utils/tensor.cuh"

template <typename T>
__global__ void op_mm_kernel(const Tensor<T> A, const Tensor<T> B, Tensor<T> C) {
  int row = blockIdx.x;
  int col = threadIdx.x;
  T temp = 0;
  for (int i = 0; i < A.w; i++) {
    temp += Index(A, row, i) * Index(B, i, col);
  }
  Index(C, row, col) = temp;
}


// This operator compute C = A@B
template <typename T>
void op_mm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) {
  assert(A.h == C.h && B.w == C.w && A.w == B.h);
  assert(A.on_device && B.on_device && C.on_device);

  // Lab-1: please complete this
  // You need to define separate kernel function(s) and launch them here
  // delete assert(0) when you are finished
  int blockSize = C.h;
  int numBlocks = C.w;
  op_mm_kernel<<<blockSize, numBlocks>>>(A, B, C);
}