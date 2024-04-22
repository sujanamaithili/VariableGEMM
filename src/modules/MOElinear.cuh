#pragma once
#include "modules/param.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_mm.cuh"
#include "modules/linear.cuh"

template<typename T>
class MOELinearLayer {
    private:
        std::vector<LinearLayer<T>> experts;
        std::vector<int> batch_splits;
        // std::vector<cudaStream_t> streams;
        int in_dim;
        int out_dim;
    public:
    MOELinearLayer(int in_dim_, int out_dim_, vector<int>& batch_splits_, bool gpu): in_dim(in_dim_), out_dim(out_dim_), batch_splits(batch_splits_){
        for(int i=0; i < batch_splits.size(); i++){
            experts.emplace_back(in_dim, out_dim, gpu);
        }
    }

    void init() {
        for(auto& expert: experts){
            expert.init_uniform();
        }
    }

    void forward(const Tensor<float> &x, Tensor<float> &y) {
        assert(x.h == y.h && y.w == out_dim);
        int start = 0;
        for(int i=0; i<experts.size(); i++){
            int batch_size = batch_splits[i];
            Tensor<T> x_sub = x.slice(start, start + batch_size, 0, in_dim);
            Tensor<T> y_sub = y.slice(start, start + batch_size, 0, out_dim);
            experts[i].forward(x_sub, y_sub);
            start += batch_size;
        }
      
    }

};