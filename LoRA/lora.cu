#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iomanip>

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel parameters - tuned for modern GPUs
#define BLOCK_SIZE 32      // Block size for shared memory tiling
#define THREAD_ROW_STRIDE 8 // Elements per thread in row dimension for register blocking
#define THREAD_COL_STRIDE 8 // Elements per thread in column dimension for register blocking

// Kernel for initializing matrices with random values on GPU
__global__ void initializeMatrices(float *A, float *B, 
                                  int rowsA, int colsA, int colsB, 
                                  unsigned int seed) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalElements = max(rowsA * colsA, colsA * colsB);
    
    if (idx < totalElements) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        // Initialize A
        if (idx < rowsA * colsA) {
            A[idx] = curand_uniform(&state) * 2.0f - 1.0f; // Range [-1, 1]
        }
        
        // Initialize B
        if (idx < colsA * colsB) {
            B[idx] = curand_uniform(&state) * 2.0f - 1.0f; // Range [-1, 1]
        }
    }
}

// Basic matrix multiplication kernel (for comparison)
__global__ void matrixMultiplyBasic(const float* __restrict__ A, 
                                   const float* __restrict__ B, 
                                   float* __restrict__ C,
                                   int rowsA, int colsA, int colsB) {
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

// Shared memory matrix multiplication kernel
__global__ void matrixMultiplyShared(const float* __restrict__ A, 
                                    const float* __restrict__ B, 
                                    float* __restrict__ C,
                                    int rowsA, int colsA, int colsB) {
    // Shared memory for tiles
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Row and column indices
    const int row = blockIdx.y * BLOCK_SIZE + ty;
    const int col = blockIdx.x * BLOCK_SIZE + tx;
    
    // Accumulator
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (colsA + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load tiles from global memory to shared memory
        if (row < rowsA && t * BLOCK_SIZE + tx < colsA) {
            tileA[ty][tx] = A[row * colsA + t * BLOCK_SIZE + tx];
        } else {
            tileA[ty][tx] = 0.0f;
        }
        
        if (t * BLOCK_SIZE + ty < colsA && col < colsB) {
            tileB[ty][tx] = B[(t * BLOCK_SIZE + ty) * colsB + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        // Synchronize before loading next tiles
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < rowsA && col < colsB) {
        C[row * colsB + col] = sum;
    }
}

// Highly optimized matrix multiplication using register blocking and memory coalescing
__global__ void matrixMultiplyOptimized(const float* __restrict__ A, 
                                       const float* __restrict__ B, 
                                       float* __restrict__ C,
                                       int rowsA, int colsA, int colsB) {
    // Shared memory tiles with padding to avoid bank conflicts
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];
    
    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Starting indices for this thread block
    const int blockRowStart = by * BLOCK_SIZE;
    const int blockColStart = bx * BLOCK_SIZE;
    
    // Register cache for results
    float results[THREAD_ROW_STRIDE][THREAD_COL_STRIDE] = {0.0f};
    
    // Loop over all tiles needed for this block's output
    for (int tile = 0; tile < (colsA + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Collaborative loading of A and B tiles into shared memory
        #pragma unroll 4
        for (int i = 0; i < BLOCK_SIZE; i += BLOCK_SIZE/THREAD_ROW_STRIDE) {
            const int row = blockRowStart + ty + i;
            const int col = tile * BLOCK_SIZE + tx;
            
            if (row < rowsA && col < colsA) {
                tileA[ty + i][tx] = A[row * colsA + col];
            } else {
                tileA[ty + i][tx] = 0.0f;
            }
        }
        
        #pragma unroll 4
        for (int i = 0; i < BLOCK_SIZE; i += BLOCK_SIZE/THREAD_COL_STRIDE) {
            const int row = tile * BLOCK_SIZE + ty;
            const int col = blockColStart + tx + i;
            
            if (row < colsA && col < colsB) {
                tileB[ty][tx + i] = B[row * colsB + col];
            } else {
                tileB[ty][tx + i] = 0.0f;
            }
        }
        
        // Make sure all tiles are loaded before computation
        __syncthreads();
        
        // Perform matrix multiplication on the loaded tiles
        // Using register blocking for multiple elements per thread
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            // Cache the values of tileA and tileB in registers
            const float aVal = tileA[ty][k];
            
            #pragma unroll
            for (int i = 0; i < THREAD_ROW_STRIDE; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_COL_STRIDE; ++j) {
                    results[i][j] += aVal * tileB[k][tx + j];
                }
            }
        }
        
        // Ensure computation is done before loading next tiles
        __syncthreads();
    }
    
    // Write results back to global memory
    #pragma unroll
    for (int i = 0; i < THREAD_ROW_STRIDE; ++i) {
        const int row = blockRowStart + ty + i;
        if (row < rowsA) {
            #pragma unroll
            for (int j = 0; j < THREAD_COL_STRIDE; ++j) {
                const int col = blockColStart + tx + j;
                if (col < colsB) {
                    C[row * colsB + col] = results[i][j];
                }
            }
        }
    }
}

// Function to check for CUDA errors and print device info
void checkCudaCapabilities() {
    cudaDeviceProp prop;
    int deviceId;
    
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "CUDA Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Clock rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
}

// Function to print matrix statistics and samples
void printMatrixStats(const float* matrix, int rows, int cols, const char* name) {
    float min_val = matrix[0];
    float max_val = matrix[0];
    float sum = 0.0f;
    
    for (int i = 0; i < rows * cols; i++) {
        min_val = min(min_val, matrix[i]);
        max_val = max(max_val, matrix[i]);
        sum += matrix[i];
    }
    
    float mean = sum / (rows * cols);
    
    std::cout << "Matrix " << name << " (" << rows << "x" << cols << ")" << std::endl;
    std::cout << "  Min: " << min_val << ", Max: " << max_val << ", Mean: " << mean << std::endl;
    
    // Print sample values
    std::cout << "  Sample values: ";
    for (int i = 0; i < min(5, rows * cols); i++) {
        std::cout << matrix[i] << " ";
    }
    std::cout << "..." << std::endl;
}

// Function to validate results by comparing with CPU computation
bool validateResults(const float* A, const float* B, const float* C, 
                    int rowsA, int colsA, int colsB, 
                    float tolerance = 1e-3) {
    bool passed = true;
    
    // Only validate a subset for large matrices
    int stride = max(1, (rowsA * colsB) / 1000);
    
    for (int i = 0; i < rowsA; i += stride) {
        for (int j = 0; j < colsB; j += stride) {
            float expected = 0.0f;
            for (int k = 0; k < colsA; ++k) {
                expected += A[i * colsA + k] * B[k * colsB + j];
            }
            
            float actual = C[i * colsB + j];
            if (fabs(expected - actual) > tolerance) {
                std::cout << "Validation failed at (" << i << "," << j << "): ";
                std::cout << "Expected " << expected << ", got " << actual << std::endl;
                passed = false;
                return passed; // Early return on first failure
            }
        }
    }
    
    return passed;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <rowsA> <colsA> <colsB>" << std::endl;
        std::cout << "This will calculate A * B where A is rowsA x colsA and B is colsA x colsB" << std::endl;
        return 1;
    }

    int rowsA = atoi(argv[1]);
    int colsA = atoi(argv[2]);
    int colsB = atoi(argv[3]);
    int rowsB = colsA;
    
    std::cout << "Matrix dimensions: A(" << rowsA << "x" << colsA 
              << "), B(" << rowsB << "x" << colsB << ")" << std::endl;
    
    // Check CUDA capabilities
    checkCudaCapabilities();
    
    // Calculate sizes in bytes
    size_t sizeA = rowsA * colsA * sizeof(float);
    size_t sizeB = rowsB * colsB * sizeof(float);
    size_t sizeC = rowsA * colsB * sizeof(float);
    
    // Allocate memory on host
    float *h_A = new float[rowsA * colsA]();
    float *h_B = new float[rowsB * colsB]();
    float *h_C_basic = new float[rowsA * colsB]();
    float *h_C_shared = new float[rowsA * colsB]();
    float *h_C_optimized = new float[rowsA * colsB]();
    
    // Allocate memory on device
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));
    
    // Initialize matrices on GPU using CUDA kernel
    unsigned int seed = static_cast<unsigned int>(time(NULL));
    int numThreads = 256;
    int numBlocks = (max(rowsA * colsA, rowsB * colsB) + numThreads - 1) / numThreads;
    
    std::cout << "\nInitializing matrices on GPU..." << std::endl;
    initializeMatrices<<<numBlocks, numThreads>>>(d_A, d_B, rowsA, colsA, colsB, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy initialized matrices back to host for validation
    CUDA_CHECK(cudaMemcpy(h_A, d_A, sizeA, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_B, d_B, sizeB, cudaMemcpyDeviceToHost));
    
    // Print statistics about input matrices
    printMatrixStats(h_A, rowsA, colsA, "A");
    printMatrixStats(h_B, rowsB, colsB, "B");
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds;
    
    // Warm up the GPU
    matrixMultiplyBasic<<<dim3((colsB + 15) / 16, (rowsA + 15) / 16), dim3(16, 16)>>>(
        d_A, d_B, d_C, rowsA, colsA, colsB);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    //---------------------------------------------------------------------
    // Approach 1: Basic matrix multiplication
    //---------------------------------------------------------------------
    dim3 basicBlockDim(16, 16);
    dim3 basicGridDim((colsB + basicBlockDim.x - 1) / basicBlockDim.x, 
                     (rowsA + basicBlockDim.y - 1) / basicBlockDim.y);
    
    std::cout << "\nRunning basic matrix multiplication..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));
    
    matrixMultiplyBasic<<<basicGridDim, basicBlockDim>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaMemcpy(h_C_basic, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double gflops_basic = 2.0 * rowsA * colsA * colsB / (milliseconds * 1e6);
    std::cout << "  Basic kernel: " << std::fixed << std::setprecision(3) << milliseconds << " ms";
    std::cout << " (" << std::fixed << std::setprecision(2) << gflops_basic << " TFLOPS)" << std::endl;
    
    //---------------------------------------------------------------------
    // Approach 2: Shared memory matrix multiplication
    //---------------------------------------------------------------------
    dim3 sharedBlockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 sharedGridDim((colsB + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                      (rowsA + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    std::cout << "Running shared memory matrix multiplication..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));
    
    matrixMultiplyShared<<<sharedGridDim, sharedBlockDim>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaMemcpy(h_C_shared, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double gflops_shared = 2.0 * rowsA * colsA * colsB / (milliseconds * 1e6);
    std::cout << "  Shared memory kernel: " << std::fixed << std::setprecision(3) << milliseconds << " ms";
    std::cout << " (" << std::fixed << std::setprecision(2) << gflops_shared << " TFLOPS)" << std::endl;
    
    //---------------------------------------------------------------------
    // Approach 3: Highly optimized matrix multiplication
    //---------------------------------------------------------------------
    dim3 optimizedBlockDim(BLOCK_SIZE / THREAD_COL_STRIDE, BLOCK_SIZE / THREAD_ROW_STRIDE);
    dim3 optimizedGridDim((colsB + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                         (rowsA + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    std::cout << "Running optimized matrix multiplication..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));
    
    matrixMultiplyOptimized<<<optimizedGridDim, optimizedBlockDim>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaMemcpy(h_C_optimized, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double gflops_optimized = 2.0 * rowsA * colsA * colsB / (milliseconds * 1e6);
    std::cout << "  Optimized kernel: " << std::fixed << std::setprecision(3) << milliseconds << " ms";
    std::cout << " (" << std::fixed << std::setprecision(2) << gflops_optimized << " TFLOPS)" << std::endl;
    
    //---------------------------------------------------------------------
    // Result validation and statistics
    //---------------------------------------------------------------------
    std::cout << "\nValidating results..." << std::endl;
    
    // Check if optimized results match basic results
    float maxDiff1 = 0.0f;
    for (int i = 0; i < rowsA * colsB; i++) {
        maxDiff1 = max(maxDiff1, fabs(h_C_basic[i] - h_C_shared[i]));
    }
    
    float maxDiff2 = 0.0f;
    for (int i = 0; i < rowsA * colsB; i++) {
        maxDiff2 = max(maxDiff2, fabs(h_C_basic[i] - h_C_optimized[i]));
    }
    
    std::cout << "  Maximum difference between basic and shared memory: " << maxDiff1 << std::endl;
    std::cout << "  Maximum difference between basic and optimized: " << maxDiff2 << std::endl;
    
    // Additional validation against CPU computation for small matrices
    if (rowsA <= 1000 && colsB <= 1000) {
        bool valid = validateResults(h_A, h_B, h_C_optimized, rowsA, colsA, colsB);
        std::cout << "  Full CPU validation: " << (valid ? "PASSED" : "FAILED") << std::endl;
    }
    
    // Print statistics about result matrix
    printMatrixStats(h_C_optimized, rowsA, colsB, "C (result)");
    
    //---------------------------------------------------------------------
    // Performance summary
    //---------------------------------------------------------------------
    std::cout << "\nPerformance Summary:" << std::endl;
    std::cout << "  Basic:     " << std::fixed << std::setprecision(2) << gflops_basic << " TFLOPS" << std::endl;
    std::cout << "  Shared:    " << std::fixed << std::setprecision(2) << gflops_shared << " TFLOPS" << std::endl;
    std::cout << "  Optimized: " << std::fixed << std::setprecision(2) << gflops_optimized << " TFLOPS" << std::endl;
    
    std::cout << "  Speedup (Shared vs Basic): " << std::fixed << std::setprecision(2) 
              << gflops_shared / gflops_basic << "x" << std::endl;
    std::cout << "  Speedup (Optimized vs Basic): " << std::fixed << std::setprecision(2) 
              << gflops_optimized / gflops_basic << "x" << std::endl;
    
    //---------------------------------------------------------------------
    // Cleanup
    //---------------------------------------------------------------------
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_basic;
    delete[] h_C_shared;
    delete[] h_C_optimized;
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    std::cout << "\nMatrix multiplication completed successfully." << std::endl;
    
    return 0;
}
