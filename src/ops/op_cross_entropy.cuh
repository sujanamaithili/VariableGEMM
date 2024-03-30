#pragma once
#include "utils/tensor.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_mm.cuh"

#define ELEMWISE_BLOCK_DIM 32

template <typename T>
__global__ void op_cross_entropy_loss_kernel(const Tensor<T> &logits, const Tensor<char> &targets, Tensor<T> &d_logits, Tensor<T> &average_loss)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < logits.h){
        // Number of class, in our case 10
        const int c = logits.w;
        const int b = logits.h;
        bool on_gpu = true;
        // Finding indices of maximum logit per batch
        Tensor<T> max_logits{logits.h, 1, on_gpu};
        op_argmax2(logits, max_logits);

        // Calculating SOFTMAX
        // from max_logits = -max_logits
        op_multiply(max_logits, -1.0, max_logits);
        // from logits = logits - max_logits
        op_add(logits, max_logits, logits);
        // from logits = exp^(logits - max_logits)
        op_exp(logits, logits);
        // sum(exp^(logits - max_logits))
        Tensor<T> sum{logits.h, 1, on_gpu};
        op_sum(logits, sum);
        // logits/sum(exp^(logits - max_logits))
        Tensor<T> softmax{logits.h, logits.w, on_gpu};
        op_divide(logits, sum, softmax);

        // -log(p_i)
        Tensor<T> negative_log_softmax{logits.h, logits.w, on_gpu};
        op_log(softmax, negative_log_softmax);
        op_multiply(negative_log_softmax, -1.0, negative_log_softmax);

        // cross entropy loss
        Tensor<T> loss{logits.h, 1, on_gpu};
        Index(loss, row, 0) = Index(negative_log_softmax, row, Index(targets, row, 0));

        // Average cross entropy loss
        op_sum(loss, average_loss);
        op_divide(average_loss, b, average_loss);

        for(int col = 0; col < c; col++){
            if(Index(targets, row, 0) == col){
                Index(d_logits, row, col) = Index(softmax, row, col) - 1;
            }
            else{
                Index(d_logits, row, col) = Index(softmax, row, col);
            }
        }

        // return Index(average_loss, 0, 0);

    }
}


//This function calculates the cross_entropy loss from the "logits" tensor for a batch of training innput
//and the batch's corresponding "target" label tensor and returns the average loss of the batch.
//It also returns the gradient of the logits tensor.
template <typename T>
T op_cross_entropy_loss(const Tensor<T> &logits, const Tensor<char> &targets,
                               Tensor<T> &d_logits)
{
    assert(logits.h == targets.h && logits.h == d_logits.h);
    assert(logits.w == d_logits.w);
    assert(targets.w == 1);

    assert(logits.on_device && targets.on_device && d_logits.on_device); 

    //Lab-2: please add your code here. 
    //You need to define separate GPU kernel function(s) and launch them here
    //In order to calculate d_logits, you should derive what its values should be 
    //symbolically.
    // return 0;
    bool on_gpu = true;
    Tensor<T> average_loss{1, 1, on_gpu};
    assert(average_loss.on_device);
    dim3 blockSize(1,ELEMWISE_BLOCK_DIM);
    dim3 numBlocks(1,(logits.h + ELEMWISE_BLOCK_DIM -1)/ELEMWISE_BLOCK_DIM);
    op_cross_entropy_loss_kernel<<<blockSize, numBlocks>>>(logits, targets, d_logits, average_loss);
    return Index(average_loss, 0, 0);
}
