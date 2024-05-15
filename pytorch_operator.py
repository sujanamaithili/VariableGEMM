import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from src/ops/op_mm.cuh import batched_gemm
from pycuda.compiler import SourceModule

mod = SourceModule(batched_gemm)

batched_sgemm = mod.get_function("batched_sgemm")

class BatchedMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        assert A.dim() == 3 and B.dim() == 3, "Inputs must be 3-dimensional tensors (batched matrices)"
        batch_size, m, k1 = A.size()
        _, k2, n = B.size()
        assert k1 == k2, "Inner dimensions must match for matrix multiplication"

        C = torch.zeros(batch_size, m, n).cuda()


        A_flat = A.contiguous().view(-1)
        B_flat = B.contiguous().view(-1)
        C_flat = C.contiguous().view(-1)

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
