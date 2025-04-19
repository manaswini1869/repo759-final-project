// Updated matrix_ops.cpp with custom CUDA matrix multiplication kernel and custom matrix-vector multiplication

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include "matrix_ops.h"
#include "utils.h"
#include "hadamard.h"
#include <iostream>

// CUDA Kernel to create Hadamard matrix directly on device
__global__ void hadamard_kernel(float* H, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) return;

    int pop = __popc(row & col);
    int sign = (pop & 1) ? -1 : 1;  // Even => 1, Odd => -1
    H[row * n + col] = static_cast<float>(sign);
}

void create_hadamard_matrix(int n, float** d_H) {
    cudaMalloc(d_H, n * n * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + 15) / 16, (n + 15) / 16);

    hadamard_kernel<<<blocksPerGrid, threadsPerBlock>>>(*d_H, n);
    cudaDeviceSynchronize();
}


// Kernel to fill C matrix
__global__ void fill_C_kernel(float* d_C, const float* d_values, const int* d_locs, int nnz, int padded_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        int row = d_locs[2 * idx];
        int col = d_locs[2 * idx + 1];
        d_C[row * padded_cols + col] = d_values[idx];
    }
}

// Custom matrix multiplication kernel
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Custom matrix-vector multiplication kernel
__global__ void matvec_kernel(const float* A, const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int k = 0; k < cols; ++k) {
            sum += A[row * cols + k] * x[k];
        }
        y[row] = sum;
    }
}

void reconstruct_C(const std::vector<float>& values, const std::vector<int>& locs,
                   int rows, int cols, int padded_rows, int padded_cols, float** d_C) {
    int nnz = values.size();

    cudaMalloc(d_C, padded_rows * padded_cols * sizeof(float));
    cudaMemset(*d_C, 0, padded_rows * padded_cols * sizeof(float));

    float* d_values;
    int* d_locs;
    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_locs, nnz * 2 * sizeof(int));

    cudaMemcpy(d_values, values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_locs, locs.data(), nnz * 2 * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (nnz + threads - 1) / threads;
    fill_C_kernel<<<blocks, threads>>>(*d_C, d_values, d_locs, nnz, padded_cols);

    cudaFree(d_values);
    cudaFree(d_locs);
}

void calculate_deltaW(float* d_H_row, float* d_C, float* d_H_col,
                      int rows, int cols, float** d_deltaW) {
    float* d_temp;
    cudaMalloc(&d_temp, rows * cols * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    // temp = H_row * C
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_H_row, d_C, d_temp, rows, cols, rows);
    cudaDeviceSynchronize();

    cudaMalloc(d_deltaW, rows * cols * sizeof(float));

    // deltaW = temp * H_col
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_temp, d_H_col, *d_deltaW, rows, cols, cols);
    cudaDeviceSynchronize();

    cudaFree(d_temp);
}

void compute_Y(float* d_deltaW, const std::string& xfile, int input_dim, int output_dim) {
    std::vector<float> X_host;
    load_vector(xfile, X_host);

    float* d_X;
    cudaMalloc(&d_X, input_dim * sizeof(float));
    cudaMemcpy(d_X, X_host.data(), X_host.size() * sizeof(float), cudaMemcpyHostToDevice);

    float* d_Y;
    cudaMalloc(&d_Y, output_dim * sizeof(float));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((output_dim + 255) / 256);

    // Custom kernel for Y = deltaW * X
    matvec_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_deltaW, d_X, d_Y, output_dim, input_dim);
    cudaDeviceSynchronize();

    std::vector<float> Y_host(output_dim);
    cudaMemcpy(Y_host.data(), d_Y, output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    std::string yfile = "outputs/Y_" + xfile.substr(xfile.find_last_of("/") + 1);
    save_vector(yfile, Y_host);

    cudaFree(d_X);
    cudaFree(d_Y);
}

int next_power_of_2(int n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}


/*CPU Implementation*/
// void create_hadamard_matrix(int n, float** d_H) {
//     std::vector<float> H_host(n * n, 1.0f);

//     for (int i = 1; i < n; i <<= 1) {
//         for (int y = 0; y < i; ++y) {
//             for (int x = 0; x < i; ++x) {
//                 H_host[(y + i) * n + x] = H_host[y * n + x];
//                 H_host[y * n + (x + i)] = H_host[y * n + x];
//                 H_host[(y + i) * n + (x + i)] = -H_host[y * n + x];
//             }
//         }
//     }

//     cudaMalloc(d_H, n * n * sizeof(float));
//     cudaMemcpy(*d_H, H_host.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
// }
