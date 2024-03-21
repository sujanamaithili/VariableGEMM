#pragma once

#include "utils/assert.cuh"
#include "utils/tensor.cuh"

#define ELEMWISE_BLOCK_DIM 32

//This operator compute C = A@B
template <typename T>
__global__ void op_mm_kernel(const Tensor<T> A, const Tensor<T> B, Tensor<T> C)
{
    __shared__ T a[ELEMWISE_BLOCK_DIM][ELEMWISE_BLOCK_DIM];
    __shared__ T b[ELEMWISE_BLOCK_DIM][ELEMWISE_BLOCK_DIM];

    // int blockx = blockIdx.x;
    // int blocky = blockIdx.y;
    int threadx = threadIdx.x;
    int thready = threadIdx.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = 0;

    for(int i=0; i < (A.w+ELEMWISE_BLOCK_DIM-1)/ELEMWISE_BLOCK_DIM; i++){
        if(row < A.h && (i*ELEMWISE_BLOCK_DIM + threadx) < A.w){
            a[thready][threadx] = Index(A, row, i*ELEMWISE_BLOCK_DIM + threadx);
        }
        else a[thready][threadx] = 0;

        if(col < B.w && (i*ELEMWISE_BLOCK_DIM + thready) < B.h){
            b[thready][threadx] = Index(B, i*ELEMWISE_BLOCK_DIM + thready,col);
        }
        else b[thready][threadx] = 0;

        __syncthreads();

        for(int k= 0; k < ELEMWISE_BLOCK_DIM; k++){
            sum += a[thready][k]*b[k][threadx];
        }
    }

}


template <typename T>
void op_mm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
{
    assert(A.h == C.h && B.w == C.w && A.w == B.h);
    assert(A.on_device && B.on_device && C.on_device);

    //Lab-1: please complete this
    //You need to define separate kernel function(s) and launch them here
    //delete assert(0) when you are finished
    // assert(0);

    dim3 blockSize(ELEMWISE_BLOCK_DIM, ELEMWISE_BLOCK_DIM);
    dim3 numBlocks((C.w + ELEMWISE_BLOCK_DIM -1)/ELEMWISE_BLOCK_DIM, (C.h + ELEMWISE_BLOCK_DIM -1)/ELEMWISE_BLOCK_DIM);
    op_mm_kernel<<<blockSize, numBlocks>>>(A, B, C);
}

