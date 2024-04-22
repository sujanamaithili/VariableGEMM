#include <getopt.h>
#include <chrono>
#include <iostream>
#include <random>
#include "modules/MOEmlp.cuh"
#include "modules/linear.cuh"

#include "utils/dataset_mnist.hh"
#include "ops/op_elemwise.cuh"
#include "ops/op_cross_entropy.cuh"

unsigned long long randgen_seed = 1;

static bool on_gpu = true;

int correct(const Tensor<float> &logits, const Tensor<char> &targets) 
{
    assert(targets.w == 1);
    Tensor<int> predictions{targets.h, targets.w, on_gpu};
    op_argmax(logits, predictions);
    Tensor<int> correct_preds{targets.h, targets.w, on_gpu};
    op_equal(predictions, targets, correct_preds);
    Tensor<int> sum_correct{1,1, on_gpu};
    op_sum(correct_preds, sum_correct);
    if (on_gpu) {
        auto tmp = sum_correct.toHost();
        return Index(tmp, 0, 0);
    }
    return Index(sum_correct, 0, 0);
}

void generate_random_moe_splits(std::vector<int>& splits, int total_batch_size, int num_splits) {
    std::mt19937 generator(randgen_seed);  
    std::vector<int> temp_splits(num_splits, 0);
    int sum = 0;

    for (int i = 0; i < num_splits - 1; ++i) {
        std::uniform_int_distribution<int> distribution(1, total_batch_size - sum - (num_splits - i));
        temp_splits[i] = distribution(generator);
        sum += temp_splits[i];
    }
    // Ensuring the splits sum to total_batch_size
    temp_splits[num_splits - 1] = total_batch_size - sum; 
    splits = std::move(temp_splits);
}


void test(int batch_size, int hidden_dim, int n_layers)
{
    MNIST mnist_test{"../data/MNIST/raw", MNIST::Mode::kTest};
    std::cout << "# of test datapoints= " << mnist_test.images.h << " feature size=" << mnist_test.images.w << std::endl;


    auto test_images = mnist_test.images;
    auto test_targets = mnist_test.targets;
    if (on_gpu) {
        test_images = test_images.toDevice();
        test_targets = test_targets.toDevice();
    }
    
    std::vector<int> layer_dims;
    for (int i = 0; i < n_layers - 1; i++)
    {
        layer_dims.push_back(hidden_dim);
    }
    layer_dims.push_back(10); // last layer's out dimension is always 10 (# of digits)

    // std::vector<int> moe_batch_splits{8, 6, 10, 8}; 
    std::vector<int> moe_batch_splits;
    generate_random_moe_splits(moe_batch_splits, batch_size, 4);


    MOEMLP<float> moeMLP{batch_size, MNIST::kImageRows * MNIST::kImageColumns, layer_dims, moe_batch_splits, on_gpu};
    moeMLP.init();

    Tensor<float> logits{batch_size, 10, on_gpu};
    int num_batches = 0, total_correct = 0;
    float total_loss = 0.0;

    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    for (int b = 0; b < test_images.h / batch_size; b++)
    {
        if ((b + 1) * batch_size > test_images.h)
        {
            break;
        }
        num_batches++;
        Tensor<float> b_images = test_images.slice(b * batch_size, (b + 1) * batch_size, 0, test_images.w);
        Tensor<char> b_targets = test_targets.slice(b * batch_size, (b + 1) * batch_size, 0, test_targets.w);

        moeMLP.forward(b_images, logits, on_gpu);
        
        total_correct += correct(logits, b_targets);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "TEST accuracy=" << total_correct / (float)(num_batches * batch_size)
              << " num_batches=" << num_batches
              << " Inference Time: " << duration.count() << " ms" << std::endl;

}

int main(int argc, char *argv[])
{
    int hidden_dim = 16;
    int n_layers = 2;
    int batch_size = 32;

    for (;;)
    {
        switch (getopt(argc, argv, "s:g:h:l:b:"))
        {
        case 's':
            randgen_seed = atoll(optarg);
            continue;
        case 'g':
            on_gpu = atoi(optarg)?true:false;
            continue;
        case 'h':
            hidden_dim = atoi(optarg);
            continue;
        case 'l':
            n_layers = atoi(optarg);
            continue;
        case 'b':
            batch_size = atoi(optarg);
            continue;
        case -1:
            break;
        }
        break;
    }
    test(batch_size, hidden_dim, n_layers);
}