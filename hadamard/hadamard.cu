#include <cnpy.h>
#include "load_ckpt.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>

// Utility: Get next power of 2
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

// Create Hadamard Matrix on Device
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

// 1D CUDA Matrix Multiplication Kernel
__global__ void matmul_kernel_1d(const float* A, const float* B, float* C, int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int row = idx / N;
    int col = idx % N;

    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <ct_mat_rows> <ct_mat_cols> <num_tokens> <threads_per_block>"
                  << " <ct_directory> <locs_directory> <x_directory>\n";
        return 1;
    }

    int ct_mat_rows_input = std::atoi(argv[1]);
    int ct_mat_cols_input = std::atoi(argv[2]);
    int num_tokens = std::atoi(argv[3]);
    int threads_per_block = std::atoi(argv[4]);
    std::string ct_directory = argv[5];
    std::string locs_directory = argv[6];
    std::string x_directory = argv[7];

    int ct_mat_rows = next_power_of_2(ct_mat_rows_input);
    int ct_mat_cols = next_power_of_2(ct_mat_cols_input);
    std::cout << "\nct_mat_rows: " << ct_mat_rows << ", ct_mat_cols: " << ct_mat_cols << "\n";

    float* h_y = new float[num_tokens * ct_mat_cols];
    float* h_D = new float[ct_mat_rows * ct_mat_cols];
    float* h_dw = new float[ct_mat_rows * ct_mat_cols];

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
    //std::cout << "\n C matrix loaded\n";

    float* d_C = nullptr;
    cudaMalloc(&d_C, ct_mat_rows * ct_mat_cols * sizeof(float));
    cudaMemcpy(d_C, ct_mat, ct_mat_rows * ct_mat_cols * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout << "\n C matrix copied into device\n";

    float* d_H_row = nullptr;
    float* d_H_col = nullptr;
    create_hadamard_matrix(ct_mat_rows, &d_H_row);
    create_hadamard_matrix(ct_mat_cols, &d_H_col);
    //std::cout << "\n Hadamard matrix created\n";

    float* d_x = nullptr;
    cudaMalloc(&d_x, x_rows * x_cols * sizeof(float));
    cudaMemcpy(d_x, x, x_rows * x_cols * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout << "\n Input X copied into device\n";

    float* d_y = nullptr;
    cudaMalloc(&d_y, x_rows * ct_mat_cols * sizeof(float));
    float* d_deltaW_trimmed;
    cudaMalloc(&d_deltaW_trimmed, ct_mat_rows * x_rows * sizeof(float));

    // CUDA events for deltaW
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // deltaW = H * C * H^T
    float* d_temp;
    float* d_deltaW = nullptr;
    cudaMalloc(&d_temp, ct_mat_rows * ct_mat_cols * sizeof(float));
    cudaMalloc(&d_deltaW, ct_mat_rows * ct_mat_cols * sizeof(float));

    int total_threads_temp = ct_mat_rows * ct_mat_cols;
    int num_blocks_temp = (total_threads_temp + threads_per_block - 1) / threads_per_block;

    cudaEventRecord(start);
    matmul_kernel_1d<<<num_blocks_temp, threads_per_block>>>(
        d_H_row, d_C, d_temp, ct_mat_rows, ct_mat_cols, ct_mat_rows);

    int total_threads_final = ct_mat_rows * ct_mat_cols;
    int num_blocks_final = (total_threads_final + threads_per_block - 1) / threads_per_block;

    matmul_kernel_1d<<<num_blocks_final, threads_per_block>>>(
        d_temp, d_H_col, d_deltaW, ct_mat_rows, ct_mat_cols, ct_mat_cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_HC = 0.0f;
    cudaEventElapsedTime(&time_HC, start, stop);

    //std::cout << "\n✅ Delta W computed using Hadamard matrices\n";
    //std::cout << "⏱ H * C kernel time: " << time_HC << " ms\n";
    //std::cout << "⏱ (H * C) * H^T kernel time: " << time_HCHT << " ms\n";

    // Trim deltaW before final Y = deltaW * X
    cudaMemcpy2D(d_deltaW_trimmed, x_rows * sizeof(float),
                 d_deltaW, ct_mat_cols * sizeof(float),
                 x_rows * sizeof(float), ct_mat_rows,
                 cudaMemcpyDeviceToDevice);

    int total_threads_Y = x_rows * ct_mat_cols;
    int num_blocks_Y = (total_threads_Y + threads_per_block - 1) / threads_per_block;

    // Time final Y = deltaW * X
    cudaEvent_t final_start, final_stop;
    cudaEventCreate(&final_start);
    cudaEventCreate(&final_stop);

    cudaEventRecord(final_start);
    matmul_kernel_1d<<<num_blocks_Y, threads_per_block>>>(
        d_deltaW_trimmed, d_x, d_y, ct_mat_rows, x_cols, x_rows);
    cudaEventRecord(final_stop);
    cudaEventSynchronize(final_stop);

    float time_Y = 0.0f;
    cudaEventElapsedTime(&time_Y, final_start, final_stop);

    //std::cout << "\n✅ Final matmul for Y done!\n";
    //std::cout << "⏱ Y = deltaW * X kernel time: " << time_Y << " ms\n";
    std::cout << "Model:" << ct_directory << "\n";
    std::cout << "Input:" << x_directory << "\n";
    std::cout << "Number of Threads per block:" << threads_per_block <<"\n";
    std::cout << "Total Time:" << time_Y + time_HC << " ms\n";

    // Cleanup
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
    cudaEventDestroy(final_start);
    cudaEventDestroy(final_stop);

    delete[] h_y;
    delete[] h_D;
    delete[] h_dw;
    delete[] ct_mat;
    delete[] ct;
    delete[] locs;
    delete[] x;

    return 0;
}
