// File: hadamard_utils.cuh
#ifndef HADAMARD_UTILS_CUH
#define HADAMARD_UTILS_CUH

#include <cuda_runtime.h>

// CUDA Matrix Multiplication Kernel Declaration
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K);

// Utility: Get next power of 2
int next_power_of_2(int n);

// Create Hadamard Matrix on Device
void create_hadamard_matrix(int n, float** d_H);

// Compute deltaW = H * C * H^T
void calculate_deltaW(float* d_H_row, float* d_C, float* d_H_col, int rows, int cols, float** d_deltaW);

#endif // HADAMARD_UTILS_CUH
