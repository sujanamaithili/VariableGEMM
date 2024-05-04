#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "ops/op_reduction.cuh"
#include "utils/assert.cuh"
#include "utils/cublas_utils.h"
#include "utils/tensor.cuh"

using data_type = float;

template <typename T>
__global__ void op_mm_kernel(const Tensor<T> A, const Tensor<T> B,
                             Tensor<T> C) {
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
void op_mm_old(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) {
   assert(A.h == C.h && B.w == C.w && A.w == B.h);
   assert(A.on_device && B.on_device && C.on_device);

   // Lab-1: please complete this
   // You need to define separate kernel function(s) and launch them here
   // delete assert(0) when you are finished
   int blockSize = C.h;
   int numBlocks = C.w;
   op_mm_kernel<<<blockSize, numBlocks>>>(A, B, C);
}

template <typename T>
__global__ void rowMajor2colMajor(const Tensor<T> A, T *d_A) {
	assert(A.on_device);
	int col = threadIdx.x;
	for (int row = 0; row < A.h; row++) {
		d_A[row + col * A.h] = Index(A, row, col);
	} 
}

template <typename T>
__global__ void colMajor2rowMajor(T *d_A, Tensor<T> A) {	
	assert(A.on_device);
	int row = threadIdx.x;
	for (int col = 0; col < A.w; col++) {
		Index(A, row, col) = d_A[row + col * A.h];
	}	
}

template <typename T>
void op_mm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) {
  assert(A.w == B.h && A.h == C.h && B.w == C.w);
 	assert(A.on_device && B.on_device && C.on_device); 
	int m = A.h, k = A.w, n = B.w;
  int lda = A.h, ldb = B.h, ldc = C.h;
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  T *alpha = new T; *alpha =  1.0;
  T *beta = new T; *beta =  0.0;
	
	cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;

  T *d_A = nullptr;
  T *d_B = nullptr;
  T *d_C = nullptr;	
	
  CUBLAS_CHECK(cublasCreate(&cublasH));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));
	
  CUDA_CHECK(
      cudaMalloc((void **)&d_A, sizeof(T) * A.h * A.w));
  CUDA_CHECK(
      cudaMalloc((void **)&d_B, sizeof(T) * B.h * B.w));
  CUDA_CHECK(
      cudaMalloc((void **)&d_C, sizeof(T) * C.h * C.w));

	rowMajor2colMajor<T><<<1, A.w>>> (A, d_A);
	rowMajor2colMajor<T><<<1, B.w>>> (B, d_B);
  
	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaDeviceSynchronize());
	CUBLAS_CHECK(cublasSgemm(cublasH, transa, transb, m, n, k, alpha, d_A, lda,
                           d_B, ldb, beta, d_C, ldc));
  CUDA_CHECK(cudaStreamSynchronize(stream));

	colMajor2rowMajor<T><<<1, C.h>>> (d_C, C);  
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
	
  CUBLAS_CHECK(cublasDestroy(cublasH));
  CUDA_CHECK(cudaStreamDestroy(stream));
	
	assert(A.on_device && B.on_device && C.on_device);
}

template <typename T>
int cublas_gemm(int m_array[], int n_array[], int k_array[], T alpha_array[],
                int lda_array[], int ldb_array[], T beta_array[],
                int ldc_array[], int group_count, int group_size[],
                const std::vector<std::vector<T>> &A_array,
                const std::vector<std::vector<T>> &B_array,
                std::vector<std::vector<T>> &C_array) {
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;

  const int gemm_count = A_array.size();
  std::vector<T *> d_A(gemm_count, nullptr);
  std::vector<T *> d_B(gemm_count, nullptr);
  std::vector<T *> d_C(gemm_count, nullptr);

  T **d_A_array = nullptr;
  T **d_B_array = nullptr;
  T **d_C_array = nullptr;

  cublasOperation_t transa_array[group_count] = {CUBLAS_OP_N, CUBLAS_OP_N};
  cublasOperation_t transb_array[group_count] = {CUBLAS_OP_N, CUBLAS_OP_N};
  int problem_idx = 0;
  for (int i = 0; i < group_count; i++) {
    printf("Group %d:\n", i);
    for (int j = 0; j < group_size[i]; j++) {
      printf("A[%d]\n", j);
      print_matrix(m_array[i], k_array[i], A_array[problem_idx].data(),
                   lda_array[i]);
      printf("=====\n");

      printf("B[%d]\n", j);
      print_matrix(k_array[i], n_array[i], B_array[problem_idx].data(),
                   ldb_array[i]);
      printf("=====\n");

      problem_idx++;
    }
    printf("\n");
  }
  /* step 1: create cublas handle, bind a stream */
  CUBLAS_CHECK(cublasCreate(&cublasH));

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  /* step 2: copy data to device */
  for (int i = 0; i < gemm_count; i++) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A[i]),
                          sizeof(T) * A_array[i].size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B[i]),
                          sizeof(T) * B_array[i].size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C[i]),
                          sizeof(T) * C_array[i].size()));
  }

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A_array),
                        sizeof(T *) * gemm_count));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B_array),
                        sizeof(T *) * gemm_count));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_array),
                        sizeof(T *) * gemm_count));

  for (int i = 0; i < gemm_count; i++) {
    CUDA_CHECK(cudaMemcpyAsync(d_A[i], A_array[i].data(),
                               sizeof(T) * A_array[i].size(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B[i], B_array[i].data(),
                               sizeof(T) * B_array[i].size(),
                               cudaMemcpyHostToDevice, stream));
  }

  CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(),
                             sizeof(T *) * gemm_count,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(),
                             sizeof(T *) * gemm_count,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(),
                             sizeof(T *) * gemm_count,
                             cudaMemcpyHostToDevice, stream));

  /* step 3: compute */
  CUBLAS_CHECK(cublasSgemmGroupedBatched(
      cublasH, transa_array, transb_array, m_array, n_array, k_array,
      alpha_array, d_A_array, lda_array, d_B_array, ldb_array, beta_array,
      d_C_array, ldc_array, group_count, group_size));

  /* step 4: copy data to host */
  for (int i = 0; i < gemm_count; i++) {
    CUDA_CHECK(cudaMemcpyAsync(C_array[i].data(), d_C[i],
                               sizeof(T) * C_array[i].size(),
                               cudaMemcpyDeviceToHost, stream));
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  problem_idx = 0;
  for (int i = 0; i < group_count; i++) {
    printf("Group %d:\n", i);
    for (int j = 0; j < group_size[i]; j++) {
      printf("C[%d]\n", j);
      print_matrix(m_array[i], n_array[i], C_array[problem_idx].data(),
                   ldc_array[i]);
      printf("=====\n");

      problem_idx++;
    }
    if (i < group_count - 1) {
      printf("\n");
    }
  }

  /* free resources */
  CUDA_CHECK(cudaFree(d_A_array));
  CUDA_CHECK(cudaFree(d_B_array));
  CUDA_CHECK(cudaFree(d_C_array));
  for (int i = 0; i < gemm_count; i++) {
    CUDA_CHECK(cudaFree(d_A[i]));
    CUDA_CHECK(cudaFree(d_B[i]));
    CUDA_CHECK(cudaFree(d_C[i]));
  }

  CUBLAS_CHECK(cublasDestroy(cublasH));

  CUDA_CHECK(cudaStreamDestroy(stream));

  // CUDA_CHECK(cudaDeviceReset());

  return EXIT_SUCCESS;
}

template <typename T>
void rowMajor_to_colMajor(const Tensor<T> &src, std::vector<T> &dest) {
  printf("allocating memory in dest\n");
	dest.resize(src.h * src.w);
	printf("copying data to destination\n");
	for (int col_idx = 0; col_idx < src.w; col_idx++) {
		for (int row_idx = 0; row_idx < src.h; row_idx++) {
			printf("idx = %d \n", col_idx * src.h + row_idx);
			printf(" val = %f\n", Index(src, row_idx, col_idx));
			dest[col_idx * src.h + row_idx] = Index(src, row_idx, col_idx);
		}
	}
  // for (int i = 0; i < src.h * src.w; i++) {
    // std::cout << src.rawp << " ";
    // dest[i] = src.rawp[i];
  // }
  std::cout << std::endl;
}

template <typename T>
void batched_gemm(const std::vector<Tensor<T>> &A_batched,
                  const std::vector<Tensor<T>> &B_batched,
                  std::vector<Tensor<T>> &C_batched) {
  printf("starting batched gemm\n");
	int gemm_count = A_batched.size();

  std::map<std::tuple<int, int, int>, std::vector<int>> size_map;
  for (int i = 0; i < gemm_count; i++) {
		printf("%d --> (%d, %d) x (%d, %d)\n", i, A_batched[i].h, A_batched[i].w, B_batched[i].h, B_batched[i].w);
    size_map[std::make_tuple(A_batched[i].h, A_batched[i].w, B_batched[i].w)]
        .push_back(i);
  }
	for (auto it : size_map) {
		printf("group\n");
		for (auto idx : it.second) {
			printf("%d %p\t", idx, A_batched[idx].rawp);
		}
		printf("\n");
	}
  int group_count = std::abs(std::distance(size_map.begin(), size_map.end()));
	printf("divided input into groups\n");
  int lda_array[group_count], ldb_array[group_count], ldc_array[group_count];
  int m_array[group_count], k_array[group_count], n_array[group_count];
  std::vector<std::vector<T>> A_array(gemm_count), B_array(gemm_count),
      C_array(gemm_count);
  int group_size[group_count];
  T alpha_array[group_count], beta_array[group_count];

  int problem_idx = 0;
  int group_idx = 0;
  for (auto it : size_map) {
		printf("arranging data for group #%d\n", group_idx);
    for (auto idx : it.second) {
			printf("converting rowMajor to colMajor of input #%d\n", problem_idx);
			printf("Tensor A = %s\n", A_batched[idx].str());
			rowMajor_to_colMajor<T>(A_batched[idx], A_array[problem_idx]);
      rowMajor_to_colMajor<T>(B_batched[idx], B_array[problem_idx]);
      rowMajor_to_colMajor<T>(C_batched[idx], C_array[problem_idx]);
      problem_idx++;
    }
    lda_array[group_idx] = std::get<0>(it.first);
    ldb_array[group_idx] = std::get<1>(it.first);
    ldc_array[group_idx] = std::get<2>(it.first);

    m_array[group_idx] = std::get<0>(it.first);
    k_array[group_idx] = ldb_array[group_idx];
    n_array[group_idx] = ldc_array[group_idx];

    group_size[group_idx] = it.second.size();
    alpha_array[group_idx] = 1.0;
    beta_array[group_idx] = 0.0;
    group_idx++;
  }
	printf("calling cublas gemm\n");
  cublas_gemm<T>(m_array, n_array, k_array, alpha_array, lda_array, ldb_array,
                 beta_array, ldc_array, group_count, group_size, A_array,
                 B_array, C_array);
	printf("reorganizing output into original order\n");
  problem_idx = 0;
  for (auto it : size_map) {
    for (auto idx : it.second) {
      for (int i = 0; i < C_array[problem_idx].size(); i++) {
        C_batched[idx].rawp[i] = C_array[problem_idx][i];
      }
      problem_idx++;
    }
  }
}
