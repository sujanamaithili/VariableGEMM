import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from pycuda.compiler import SourceModule

# CUDA kernel for matrix multiplication using cuBLAS sgemm
cuda_kernel = """
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

extern "C" {
__global__ void batched_sgemm(float *A, float *B, float *C, int m, int n, int k, int batch_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < batch_size) {
        cublasHandle_t handle;
        cublasCreate(&handle);

        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B + tid * n * k, n, A + tid * m * k, k, &beta, C + tid * n * m, n);

        cublasDestroy(handle);
    }
}
}
"""

# Compile the CUDA kernel
mod = SourceModule(cuda_kernel)

# Get the compiled function
batched_sgemm = mod.get_function("batched_sgemm")

# Define the PyTorch operator
class BatchedMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        assert A.dim() == 3 and B.dim() == 3, "Inputs must be 3-dimensional tensors (batched matrices)"
        batch_size, m, k1 = A.size()
        _, k2, n = B.size()
        assert k1 == k2, "Inner dimensions must match for matrix multiplication"

        C = torch.zeros(batch_size, m, n).cuda()

        # Flatten the tensors for passing to CUDA kernel
        A_flat = A.contiguous().view(-1)
        B_flat = B.contiguous().view(-1)
        C_flat = C.contiguous().view(-1)

        # Call the CUDA kernel
        block_size = 256
        grid_size = (batch_size + block_size - 1) // block_size
        batched_sgemm(cuda.In(A_flat), cuda.In(B_flat), cuda.Out(C_flat), np.int32(m), np.int32(n), np.int32(k1), np.int32(batch_size), block=(block_size, 1, 1), grid=(grid_size, 1))

        ctx.save_for_backward(A, B)
        return C

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_B = None

        if ctx.needs_input_grad[0]:
            grad_A = torch.matmul(grad_output, B.transpose(1, 2))

        if ctx.needs_input_grad[1]:
            grad_B = torch.matmul(A.transpose(1, 2), grad_output)

        return grad_A, grad_B
