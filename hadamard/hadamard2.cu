#include <cnpy.h>
#include "load_ckpt.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_kernel_shared(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

int next_power_of_2(int n) {
    if (n <= 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

void create_hadamard_matrix(int n, float** d_H) {
    std::vector<float> H_host(n * n, 1.0f);
    for (int i = 1; i < n; i <<= 1) {
        for (int y = 0; y < i; ++y) {
            for (int x = 0; x < i; ++x) {
                H_host[(y + i) * n + x] = H_host[y * n + x];
                H_host[y * n + (x + i)] = H_host[y * n + x];
                H_host[(y + i) * n + (x + i)] = -H_host[y * n + x];
            }
        }
    }
    cudaMalloc(d_H, n * n * sizeof(float));
    cudaMemcpy(*d_H, H_host.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <ct_mat_rows> <ct_mat_cols> <num_tokens> <threads_per_block> <ct_directory> <locs_directory> <x_directory>\n";
        return 1;
    }

    int ct_mat_rows_input = std::atoi(argv[1]);
    int ct_mat_cols_input = std::atoi(argv[2]);
    int num_tokens = std::atoi(argv[3]);
    std::string ct_directory = argv[5];
    std::string locs_directory = argv[6];
    std::string x_directory = argv[7];

    int ct_mat_rows = next_power_of_2(ct_mat_rows_input);
    int ct_mat_cols = next_power_of_2(ct_mat_cols_input);

    float* ct = nullptr;
    auto [ct_rows, ct_cols] = load_ckpt_float(ct_directory, ct);

    int* locs = nullptr;
    auto [locs_rows, locs_cols] = load_ckpt_int(locs_directory, locs);

    float* x = nullptr;
    auto [x_rows, x_cols] = load_ckpt_float(x_directory, x);

    float* ct_mat = new float[ct_mat_rows * ct_mat_cols]();
    for (size_t i = 0; i < locs_cols; ++i) {
        int row = locs[i];
        int col = locs[locs_cols + i];
        ct_mat[row * ct_mat_cols + col] = ct[i];
    }

    float *d_C, *d_H_row, *d_H_col, *d_temp, *d_deltaW, *d_x, *d_y, *d_deltaW_trimmed;
    cudaMalloc(&d_C, ct_mat_rows * ct_mat_cols * sizeof(float));
    cudaMemcpy(d_C, ct_mat, ct_mat_rows * ct_mat_cols * sizeof(float), cudaMemcpyHostToDevice);
    create_hadamard_matrix(ct_mat_rows, &d_H_row);
    create_hadamard_matrix(ct_mat_cols, &d_H_col);
    cudaMalloc(&d_x, x_rows * x_cols * sizeof(float));
    cudaMemcpy(d_x, x, x_rows * x_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, x_rows * ct_mat_cols * sizeof(float));
    cudaMalloc(&d_temp, ct_mat_rows * ct_mat_cols * sizeof(float));
    cudaMalloc(&d_deltaW, ct_mat_rows * ct_mat_cols * sizeof(float));
    cudaMalloc(&d_deltaW_trimmed, ct_mat_rows * x_rows * sizeof(float));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim_HC((ct_mat_cols + TILE_SIZE - 1) / TILE_SIZE, (ct_mat_rows + TILE_SIZE - 1) / TILE_SIZE);
    dim3 gridDim_HT((ct_mat_cols + TILE_SIZE - 1) / TILE_SIZE, (ct_mat_rows + TILE_SIZE - 1) / TILE_SIZE);
    dim3 gridDim_Y((ct_mat_cols + TILE_SIZE - 1) / TILE_SIZE, (x_rows + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_kernel_shared<<<gridDim_HC, blockDim>>>(d_H_row, d_C, d_temp, ct_mat_rows, ct_mat_cols, ct_mat_rows);
    matmul_kernel_shared<<<gridDim_HT, blockDim>>>(d_temp, d_H_col, d_deltaW, ct_mat_rows, ct_mat_cols, ct_mat_cols);

    cudaMemcpy2D(d_deltaW_trimmed, x_rows * sizeof(float), d_deltaW, ct_mat_cols * sizeof(float), x_rows * sizeof(float), ct_mat_rows, cudaMemcpyDeviceToDevice);
    matmul_kernel_shared<<<gridDim_Y, blockDim>>>(d_deltaW_trimmed, d_x, d_y, ct_mat_rows, x_cols, x_rows);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Model: " << ct_directory << "\n";
    std::cout << "Input: " << x_directory << "\n";
    std::cout << "Block size: " << TILE_SIZE << "\n";
    std::cout << "Total CUDA execution time: " << milliseconds << " ms\n";

    cudaFree(d_H_row);
    cudaFree(d_H_col);
    cudaFree(d_C);
    cudaFree(d_temp);
    cudaFree(d_deltaW);
    cudaFree(d_deltaW_trimmed);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] ct_mat;
    delete[] ct;
    delete[] locs;
    delete[] x;

    return 0;
}
