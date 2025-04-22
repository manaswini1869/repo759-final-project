#include "matmul.cuh"
#include <cstddef>
#include <cstdio>


// Does C = A * B
__global__ void frame_matmul_kernel(const float* A, const float* B, float* C, size_t m, size_t k, size_t n)
{
    // Calculate the row and column index for the element to compute
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = idx / n; // Row index
    size_t col = idx % n; // Column index

    // Check if the indices are within bounds
    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < k; ++i)
        {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Does C = A * B^T
__global__ void frame_matmul_transpose_kernel(const float* A, const float* B, float* C, size_t m, size_t k, size_t n)
{
    // Calculate the row and column index for the element to compute
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = idx / n; // Row index
    size_t col = idx % n; // Column index

    // Check if the indices are within bounds
    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < k; ++i)
        {
            sum += A[row * k + i] * B[col * k + i];
        }
        C[row * n + col] = sum;
    }
}


void frame_compute_dw(const float* tff_m, const float* tff_n, const float* ct_mat, float* D, float* result, size_t m, size_t n, unsigned int threads_per_block)
{
    size_t num_blocks = (m * n + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    frame_matmul_kernel<<<num_blocks, threads_per_block>>>(tff_m, ct_mat, D, m, m, n);

    // launch the kernel for the right multiplication
    frame_matmul_transpose_kernel<<<num_blocks, threads_per_block>>>(D, tff_n, result, m, n, n);
}


void frame_compute_y(const float* x, const float* dw, float* y, size_t n, size_t m, size_t d, unsigned int threads_per_block)
{
    size_t num_blocks = (n * d + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    frame_matmul_kernel<<<num_blocks, threads_per_block>>>(x, dw, y, n, m, d);
}