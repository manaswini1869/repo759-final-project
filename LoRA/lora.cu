#include <iostream>
#include <cuda.h>
#include <random>
#include "vscale.cuh"
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; ++k) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " <rowsA> <colsA> <colsB>" << endl;
        return 1;
    }

    // Matrix dimensions
    int rowsA = atoi(argv[1]);
    int colsA = atoi(argv[2]); // also rowsB
    int colsB = atoi(argv[3]);
    int rowsB = colsA;
    
    // Calculate total elements for each matrix
    int sizeA = rowsA * colsA;
    int sizeB = rowsB * colsB;
    int sizeC = rowsA * colsB;
    
    // Allocate memory for matrices on the host
    float* A = new float[sizeA];
    float* B = new float[sizeB];
    float* C = new float[sizeC];

    // Initialize random number generators
    default_random_engine gen;
    uniform_real_distribution<float> distribution(-10.0, 10.0);

    // Initialize matrices with random values
    for (int i = 0; i < sizeA; i++) {
        A[i] = distribution(gen);
    }
    
    for (int i = 0; i < sizeB; i++) {
        B[i] = distribution(gen);
    }

    // Allocate memory on the device
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, sizeA * sizeof(float));
    cudaMalloc((void**)&d_B, sizeB * sizeof(float));
    cudaMalloc((void**)&d_C, sizeC * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB * sizeof(float), cudaMemcpyHostToDevice);

    // Set up the execution configuration
    // Using 16x16 block size which is common for matrix operations
    dim3 blockDim(16, 16);
    dim3 gridDim((colsB + blockDim.x - 1) / blockDim.x, 
                 (rowsA + blockDim.y - 1) / blockDim.y);

    // CUDA events for timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Start timing
    cudaEventRecord(start);
    
    // Launch the kernel
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    // End timing
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    cout << "Execution time: " << milliseconds << " ms" << endl;

    // Copy the result back to host
    cudaMemcpy(C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    // Print sample results (first element, one from middle, and last element)
    cout << "Sample results:" << endl;
    cout << "C[0][0] = " << C[0] << endl;
    cout << "C[" << rowsA/2 << "][" << colsB/2 << "] = " << C[(rowsA/2) * colsB + (colsB/2)] << endl;
    cout << "C[" << rowsA-1 << "][" << colsB-1 << "] = " << C[sizeC - 1] << endl;

    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}
