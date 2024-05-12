#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils/cublas_utils.h"
#include "utils/tensor.cuh"

#define CHECK_CUSPARSE(func)																																			\
		do {																																													\
			cusparseStatus_t status = (func);																														\
			if (status != CUSPARSE_STATUS_SUCCESS) {																										\
				std::printf("cublas error %d at %s:%d\n", 																								\
								status, __FILE__, __LINE__);																											\
				throw std::runtime_error("cublas error\n");																								\
			}																																														\
		} while (0)																																										

template <typename T>
__global__ void vjoin_matrices_kernel(const std::vector<Tensor<T>> A_arr, T *dest) {
	int mat_idx = threadIdx.x;
	int start_idx = 0;
	for (int i = 0; i < mat_idx; i++) {
		start_idx += A_arr[i].h * A_arr[i].w;
	}
	
	for (int i = 0; i < A_arr[mat_idx].h * A_arr[mat_idx].w; i++) {
		dest[start_idx + i] = Index(A_arr[mat_idx], i / A_arr[mat_idx].w, i % A_arr[mat_idx].w);
	}
}

template <typename T>
__global__ void hjoin_matrices_kernel(const std::vector<Tensor<T>> A_arr, T *dest, const int n_tot) {
	int row = threadIdx.x;
	int idx = n_tot * row;

	for (int i = 0; i < A_arr.size(); i++) {
		for (int j = 0; j < A_arr[i].w; j++) {
			dest[idx] = Index(A_arr[i], row, j);
			idx++;
		}
	}
}

template <typename T>
__global void extract_tensors(const T *dC_values, std::vector<Tensor<T>> C) {
	int mat_idx = threadIdx.x;
	int start_idx = 0;
	for (int i = 0; i < mat_idx; i++) {
		start_idx += C[mat_idx].h * C[mat_idx].w;
	}

	for (int i = 0; i < C[mat_idx].h; i++) {
		for (int j = 0; j < C[mat_idx].w; j++) {
			Index(C[mat_idx], i, j) = dC_values[start_idx + i * C[mat_idx].w + j];
		}
	}
}

template <typename T>
void op_sdmm (const std::vector<Tensor<T>> &A, const std::vector<Tensor<T>> &B, std::vector<Tensor<T>> &C) {
	int gemm_count = A.size();
	for (int idx = 0; idx < gemm_count; idx++) {
		assert(A[idx].on_device && B[idx].on_device && C[idx].on_device);
		assert(A[idx].w == B[idx].h && A[idx].h == C[idx].h && B[idx].w == C[idx].w);
	}	

	int m_tot = 0;
	int k = 0;
	int n_tot = 0;
	int C_nnz = 0;
	for (int idx = 0; idx < gemm_count; idx++) {
		m_tot += A[idx].h;
		n_tot += B[idx].w;
		C_nnz += A[idx].h * B[idx].w;
	}
	k = A[0].w;
	
	int *hC_offsets, *hC_columns, *hC_values;
	hC_offsets = (int *)malloc((m_tot + 1) * sizeof(int));
	hC_columns = (int *)malloc(C_nnz * sizeof(int));
	hC_values = (T *)malloc(C_nnz * sizeof(T));

	hC_offsets[0] = 0;
	int idx = 1;
	for (int i = 0; i < gemm_count; i++) {
		for (int j = 0 ; j < A[i].h; j++) {
			hC_offsets[idx] = hC_offsets[idx - 1] + B[i].w;
			idx++;
		}
	}
	idx = 0;
	int offset = 0;
	for (int i = 0; i < gemm_count; i++) {
		for (int j = 0; j < A[i].h; j++) {
			for (int l = 0; l < B[i].w; l++) {
				hC_columns[idx] = offset + l;
			}
		}
		offset += B[i].w;
	}
	memset(hC_values, 0, C_nnz);
	
	int *dC_offsets, *dC_columns;
	T *dA_dense, *dB_dense, *dC_values;
	int lda = k, ldb = n_tot;
		
	CUDA_CHECK( cudaMalloc((void **) &dA_dense, m_tot * k * sizeof(T)) );
	CUDA_CHECK( cudaMalloc((void **) &dB_dense, k * n_tot * sizeof(T)) );
	CUDA_CHECK( cudaMalloc((void **) &dC_offsets, (m_tot + 1) * sizeof(int)) );
	CUDA_CHECK( cudaMalloc((void **) &dC_columns, C_nnz * sizeof(int)) );
	CUDA_CHECK( cudaMalloc((void **) &dC_values, C_nnz * sizeof(T)) );

	// join A tensors verticaly into dA_dense
	vjoin_matrices_kernel<<<1, gemm_count>>>(A, dA_dense);
	// join B tensor horizontally into dB_dense
	hjoin_matrices_kernel<<<1, k>>>(B, dB_dense, n_tot);
	// set dC_offsets and dC_columns	
	CUDA_CHECK( cudaMemcpy(dC_offsets, hC_offsets, (m_tot + 1) * sizeof(int), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(dC_columns, hC_columns, C_nnz * sizeof(int), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(dC_values, hC_values, C_nnz * sizeof(T), cudaMemcpyHostToDevice) );

	// cuSPARSE APIs
	cusparseHandle_t 			handle = NULL;
	cusparseDnMatDescr_t 	matA, matB;	
	cusparseSpMatDescr_t 	matC;
	void*									dBuffer = NULL;
	size_t								bufferSize = 0;

	CHECK_CUSPARSE( cusparseCreate(&handle) );
	CHECK_CUSPARSE( cusparseCreateDnMat(&matA, m_tot, k, lda, dA_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW) );
	CHECK_CUSPARSE( cusparseCreateDnMat(&matB, k, n_tot, ldb, dB_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW) );
	CHECK_CUSPARSE( cusparseCreateSpMat(&matC, m_tot, n_tot, C_nnz, dC_offsets, dC_columns, dC_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
	
	CHECK_CUSPARSE( cusparseSDDMM_bufferSize(
																handle,
																CUSPARSE_OPERATION_NON_TRANSPOSE,
																CUSPARSE_OPERATION_NON_TRANSPOSE,
																&alpha, matA, matB, &beta, matC, CUDA_R_32F,
																CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize) );

	CUDA_CHECK( cudaMalloc(&dBuffer, bufferSize) );
	
	// execute preprocess
	CHECK_CUSPARSE( cusparseSDDMM_preprocess(
																handle,
																CUSPARSE_OPERATION_NON_TRANSPOSE,
																CUSPARSE_OPERATION_NON_TRANSPOSE,
																&alpha, matA, matB, &beta, matC, CUDA_R_32F,
																CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer) );
	
	// execute SpMM
	CHECK_CUSPARSE( cusparseSDDMM(handle,
																CUSPARSE_OPERATION_NON_TRANSPOSE,
																CUSPARSE_OPERATION_NON_TRANSPOSE,
																&alpha, matA, matB, &beta, matC, CUDA_R_32F,
																CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer) );
	
	CHECK_CUSPARSE( cusparseDestroyDnMat(matA) );
	CHECK_CUSPARSE( cusparseDestroyDnMat(matB) );
	CHECK_CUSPARSE( cusparseDestroySpMat(matC) );
	CHECK_CUSPARSE( cusparseDestroy(handle) );

// 	CHECK_CUDA( cudaMemcpy(hC_values, dC_values, C_nnz * sizeof(float), cudaMemcpyDeviceToHost) );
	
	extract_tensors_kernel<<<1, gemm_count>>>(dC_values, C);

	// free device memory
	CHECK_CUDA( cudaFree(dBuffer) )
  CHECK_CUDA( cudaFree(dA_dense) )
  CHECK_CUDA( cudaFree(dB_dense) )
  CHECK_CUDA( cudaFree(dC_offsets) )
  CHECK_CUDA( cudaFree(dC_columns) )
  CHECK_CUDA( cudaFree(dC_values) )
	
	// free host memory
	delete[] hC_offsets;
	delete[] hC_columns;
	delete[] hC_values;

}
