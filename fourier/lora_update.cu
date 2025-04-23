#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "load_ckpt.h"
#include "cnpy.h"

// Thread block size
#define BLOCK_SIZE 32

// CUDA kernel for matrix multiplication (optimized with shared memory)
__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int K, int N) {
    // Block index
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread index
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // Shared memory for tile of input matrices
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of C
    float sum = 0.0f;

    // Loop over all tiles
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Load one tile of A and B into shared memory
        int aRow = blockRow * BLOCK_SIZE + threadRow;
        int aCol = t * BLOCK_SIZE + threadCol;
        int bRow = t * BLOCK_SIZE + threadRow;
        int bCol = blockCol * BLOCK_SIZE + threadCol;

        // Boundary check
        if (aRow < M && aCol < K)
            As[threadRow][threadCol] = A[aRow * K + aCol];
        else
            As[threadRow][threadCol] = 0.0f;

        if (bRow < K && bCol < N)
            Bs[threadRow][threadCol] = B[bRow * N + bCol];
        else
            Bs[threadRow][threadCol] = 0.0f;

        __syncthreads(); // Wait for all threads to load data

        // Calculate partial dot product
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadRow][k] * Bs[k][threadCol];
        }

        __syncthreads(); // Wait for all threads to finish using the tile
    }

    // Write result
    int cRow = blockRow * BLOCK_SIZE + threadRow;
    int cCol = blockCol * BLOCK_SIZE + threadCol;
    if (cRow < M && cCol < N) {
        C[cRow * N + cCol] = sum;
    }
}
// CUDA Error checking
// #define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
// void check(cudaError_t result, char const *const func, const char *const file, int const line) {
//     if (result) {
//         printf("CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
//                static_cast<unsigned int>(result), cudaGetErrorString(result), func);
//         exit(EXIT_FAILURE);
//     }
// }

int main(int argc, char* argv[]) {
    // Check if correct number of arguments are provided
 /*   if (argc != 5) {
        printf("Usage: %s <M> <K> <N> <V>\n", argv[0]);
        printf("Where:\n");
        printf("  M = number of rows in matrix A\n");
        printf("  K = number of columns in matrix A / rows in matrix B\n");
        printf("  N = number of columns in matrix B / rows in matrix X\n");
        printf("  V = number of columns in matrix X\n");
        return 1;
    }*/

    srand(time(NULL));

    float *h_A = nullptr;
    float *h_B = nullptr;
    float *h_X = nullptr;

    std::string A_dir = "./lora/gemma-2-2b-loraA.npy";
    std::string B_dir = "./lora/gemma-2-2b-loraB.npy";
    std::string X_dir = "./inputs/x_1024.npy";

    auto [A_rows, A_cols] =  load_ckpt_float(A_dir, h_A);
    auto [B_rows,B_cols] = load_ckpt_float(B_dir,h_B);
    auto [X_rows,X_cols] = load_ckpt_float(X_dir,h_X);

     size_t bytes_A = A_rows * A_cols * sizeof(float);
    size_t bytes_B = B_rows * B_cols * sizeof(float);
    size_t bytes_W = A_rows * B_cols * sizeof(float);
    size_t bytes_X = X_rows * X_cols * sizeof(float);
    size_t bytes_Y = A_rows * X_cols * sizeof(float);

    float *h_W = (float*)malloc(bytes_W);
	float *h_Y = (float*)malloc(bytes_Y);

    // Allocate device memory
    float *d_A, *d_B, *d_W, *d_X, *d_Y;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_W, bytes_W);
    cudaMalloc(&d_X, bytes_X);
    cudaMalloc(&d_Y, bytes_Y);

    // Copy host to device
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, bytes_X, cudaMemcpyHostToDevice);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid1((B_cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (A_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimGrid2((X_cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (A_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("\nRunning Calculation: Y = (A.B).X\n");
    printf("----------------------------------\n");

    // Start total timing
    cudaEventRecord(start);

    // Step 1: W = A.B
    cudaEventRecord(start);
    matrixMulKernel<<<dimGrid1, dimBlock>>>(d_A, d_B, d_W, A_rows, A_cols, B_cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for W = A.B: %f ms\n", milliseconds);
    float time_W = milliseconds;

    // Step 2: Y = W.X
    cudaEventRecord(start);
    matrixMulKernel<<<dimGrid2, dimBlock>>>(d_W, d_X, d_Y, A_rows, B_cols, X_cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for Y = W.X: %f ms\n", milliseconds);
    float time_Y = milliseconds;

    // End total timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total time for Y = (A.B).X: %f ms\n", time_W + time_Y);

    // Copy result back to host
    cudaMemcpy(h_Y, d_Y, bytes_Y, cudaMemcpyDeviceToHost);

    // Print first and last elements of results
    printf("\nResults:\n");
    printf("Y[0][0] = %f (First element)\n", h_Y[0]);
    //printf("Y[%d][%d] = %f (Last element)\n", A_rows-1, X_cols-1, h_Y[(A_rows*X_cols)-1]);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_W);
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_A);
    free(h_B);
    free(h_W);
    free(h_X);
    free(h_Y);

    return 0;
}
