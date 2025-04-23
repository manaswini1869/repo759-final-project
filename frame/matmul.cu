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

// Does C = A^T * B
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
            sum += A[i * m + row] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}


// Does C = block(A) * B
__global__ void frame_block_matmul_kernel(const float* b_A, const float* B, float* C, size_t k_m, size_t k_n, size_t l_m, size_t l_n, size_t rows, size_t cols)
{
    // Calculate the row and column index for the element to compute
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = idx / cols; // Row index
    size_t col = idx % cols; // Column index

    size_t num_blocks = (k_m < k_n) ? k_m : k_n;

    // Check if the indices are within bounds
    if (row < rows && col < cols)
    {
        float sum = 0.0f;

        int block_id = row / l_m;
        if (block_id < num_blocks)
        {
            int block_row = row % l_m;
            for (size_t i = 0; i < l_n; ++i)
            {
                sum += b_A[block_id * l_m * l_n + block_row * l_n + i] * B[(block_id * l_m + i) * cols + col];
            }
        }
        C[row * cols + col] = sum;
    }
}


void frame_compute_dw(const float* tff_m, const float* tff_n, const float* ct_mat, float* D, float* result, size_t m, size_t n, unsigned int threads_per_block)
{
    size_t num_blocks = (m * n + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    frame_matmul_transpose_kernel<<<num_blocks, threads_per_block>>>(tff_m, ct_mat, D, m, m, n);

    // launch the kernel for the right multiplication
    frame_matmul_kernel<<<num_blocks, threads_per_block>>>(D, tff_n, result, m, n, n);
}


void frame_compute_y(const float* x, const float* dw, float* y, size_t n, size_t m, size_t d, unsigned int threads_per_block)
{
    size_t num_blocks = (n * d + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    frame_matmul_kernel<<<num_blocks, threads_per_block>>>(x, dw, y, n, m, d);
}




void frame_compute_y_2( const float* tff_m,
                        const float* tff_n,
                        const float* bct,
                        const float* x,
                        float* D1,
                        float* D2,
                        float* y,
                        size_t num_tokens,
                        size_t ct_mat_rows,
                        size_t ct_mat_cols,
                        size_t k_m, size_t k_n, size_t l_m, size_t l_n,
                        unsigned int threads_per_block)
{

    size_t num_blocks_d1 = (num_tokens * ct_mat_rows + threads_per_block - 1) / threads_per_block;
    frame_matmul_transpose_kernel<<<num_blocks_d1, threads_per_block>>>(x, tff_m, D1, num_tokens, ct_mat_rows, ct_mat_rows);

    size_t num_blocks_d2 = (ct_mat_rows * ct_mat_cols + threads_per_block - 1) / threads_per_block;
    frame_block_matmul_kernel<<<num_blocks_d2, threads_per_block>>>(bct, tff_n, D2, k_m, k_n, l_m, l_n, ct_mat_rows, ct_mat_cols);

    size_t num_blocks_y = (num_tokens * ct_mat_cols + threads_per_block - 1) / threads_per_block;
    frame_matmul_kernel<<<num_blocks_y, threads_per_block>>>(D1, D2, y, num_tokens, ct_mat_rows, ct_mat_cols);

}