#pragma once
#include "utils/tensor.cuh"

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
    return 0;

}
