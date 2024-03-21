#pragma once

#include "utils/assert.cuh"
#include "utils/tensor.cuh"

//This operator compute C = A@B
template <typename T>
void op_mm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
{
    assert(A.h == C.h && B.w == C.w && A.w == B.h);
    assert(A.on_device && B.on_device && C.on_device);

    //Lab-1: please complete this
    //You need to define separate kernel function(s) and launch them here
    //delete assert(0) when you are finished
    assert(0);
}
