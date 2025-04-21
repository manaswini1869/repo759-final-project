#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

// Function to check CUDA errors
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernel for matrix multiplication C = A * B
__global__ void matrixMul(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

// CUDA kernel to initialize matrices with random values
__global__ void initializeRandom(float *data, int size, unsigned long seed, float min, float max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = min + curand_uniform(&state) * (max - min);
    }
}

int main(int argc, char *argv[]) {
    // Default dimensions if not provided
    int M = 1024;  // A rows
    int K = 512;   // A columns, B rows
    int N = 256;   // B columns
    int P = 2048;  // X rows (must equal N for matrix multiplication)
                   // X is of shape (P x N), final output Y will be (P x M)
    
    // Parse command line arguments for matrix dimensions
    if (argc >= 5) {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
        P = atoi(argv[4]);
        
        // Validate dimensions
        if (M <= 0 || K <= 0 || N <= 0 || P <= 0) {
            fprintf(stderr, "Error: Matrix dimensions must be positive integers.\n");
            exit(EXIT_FAILURE);
        }
    } else if (argc > 1 && argc < 5) {
        fprintf(stderr, "Usage: %s [M K N P]\n", argv[0]);
        fprintf(stderr, "  M: rows of matrix A\n");
        fprintf(stderr, "  K: columns of matrix A, rows of matrix B\n");
        fprintf(stderr, "  N: columns of matrix B\n");
        fprintf(stderr, "  P: rows of matrix X (X has N columns)\n");
        fprintf(stderr, "Using default values: M=%d, K=%d, N=%d, P=%d\n", M, K, N, P);
    }
    
    printf("Matrix dimensions:\n");
    printf("A: %d x %d\n", M, K);
    printf("B: %d x %d\n", K, N);
    printf("X: %d x %d\n", P, N);
    printf("Y: %d x %d\n", P, M);
    
    // Host matrices
    float *h_A, *h_B, *h_X, *h_Y;
    
    // Device matrices
    float *d_A, *d_B, *d_X, *d_XB, *d_Y;
    
    // Allocate host memory
    h_A = (float*)malloc(M * K * sizeof(float));
    h_B = (float*)malloc(K * N * sizeof(float));
    h_X = (float*)malloc(P * N * sizeof(float));
    h_Y = (float*)malloc(P * M * sizeof(float));
    
    if (!h_A || !h_B || !h_X || !h_Y) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_X, P * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_XB, P * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Y, P * M * sizeof(float)));
    
    // Define thread block and grid dimensions for initialization
    dim3 initBlockSize(256);
    dim3 initGridSizeA((M * K + initBlockSize.x - 1) / initBlockSize.x);
    dim3 initGridSizeB((K * N + initBlockSize.x - 1) / initBlockSize.x);
    dim3 initGridSizeX((P * N + initBlockSize.x - 1) / initBlockSize.x);
    
    // Initialize matrices with random values on the device
    initializeRandom<<<initGridSizeA, initBlockSize>>>(d_A, M * K, 12345, 0.0f, 1.0f);
    initializeRandom<<<initGridSizeB, initBlockSize>>>(d_B, K * N, 67890, 0.0f, 1.0f);
    initializeRandom<<<initGridSizeX, initBlockSize>>>(d_X, P * N, 54321, -1.0f, 1.0f);
    
    // Check for initialization errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Define optimal thread block size based on device capabilities
    int blockDim = 16; // Default, can be adjusted
    
    // Get device properties to optimize thread block size
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    
    // Can adjust blockDim based on device properties for better performance
    if (deviceProp.maxThreadsPerBlock >= 1024) {
        blockDim = 32; // Use larger block size on more capable hardware
    }
    
    dim3 blockSize(blockDim, blockDim);
    dim3 gridSizeXB((K + blockSize.x - 1) / blockSize.x, (P + blockSize.y - 1) / blockSize.y);
    dim3 gridSizeY((M + blockSize.x - 1) / blockSize.x, (P + blockSize.y - 1) / blockSize.y);
    
    // Record start time
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    // First compute X*B (P×N * N×K = P×K)
    matrixMul<<<gridSizeXB, blockSize>>>(d_X, d_B, d_XB, P, N, K);
    
    // Then compute (X*B)*A (P×K * K×M = P×M)
    matrixMul<<<gridSizeY, blockSize>>>(d_XB, d_A, d_Y, P, K, M);
    
    // Record end time
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Wait for GPU to finish
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    printf("Matrix multiplication completed in %.2f ms\n", milliseconds);
    
    // Copy result from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_Y, d_Y, P * M * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print some values from the result for verification
    printf("First few values of Y (output matrix):\n");
    int preview_size = 5;
    for (int i = 0; i < preview_size && i < P; i++) {
        for (int j = 0; j < preview_size && j < M; j++) {
            printf("%f ", h_Y[i * M + j]);
        }
        printf("\n");
    }
    
   
    
    // Clean up
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_X));
    CHECK_CUDA_ERROR(cudaFree(d_XB));
    CHECK_CUDA_ERROR(cudaFree(d_Y));
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_X);
    free(h_Y);
    
    return 0;
}
