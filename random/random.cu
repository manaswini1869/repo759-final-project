#include <iostream>
#include <cuda.h>
#include <random>
#include <cstdlib>
#include <ctime>

using namespace std;

// CUDA kernel for diagonal matrix multiplication with A: A * d
__global__ void multiplyWithDiagonal(float* A, float* d, float* temp, int rowsA, int colsA) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rowsA && col < colsA) {
        // For each element in A, multiply by the corresponding diagonal element in d
        temp[row * colsA + col] = A[row * colsA + col] * d[col];
    }
}

// CUDA kernel for matrix multiplication: temp * B
__global__ void matrixMultiply(float* temp, float* B, float* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; ++k) {
            sum += temp[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " <rowsA> <colsA> <colsB>" << endl;
        cout << "This will calculate A * d * B where d is a diagonal matrix of size colsA x colsA" << endl;
        return 1;
    }

    // Matrix dimensions
    int rowsA = atoi(argv[1]);
    int colsA = atoi(argv[2]); // colsA is also the size of diagonal matrix d
    int colsB = atoi(argv[3]);
    int rowsB = colsA; // Number of rows in B must equal number of columns in A
    
    // Calculate total elements for each matrix
    int sizeA = rowsA * colsA;
    int sized = colsA; // Diagonal matrix only needs to store diagonal elements
    int sizeB = rowsB * colsB;
    int sizeC = rowsA * colsB;
    int sizeTemp = sizeA;  // Temporary result of A*d
    
    // Allocate memory for matrices on the host
    float* A = new float[sizeA];
    float* d = new float[sized];
    float* B = new float[sizeB];
    float* C = new float[sizeC];
    float* temp = new float[sizeTemp];

    // Initialize random number generators
    random_device rd;
    mt19937 gen(rd());
    
    // For A and B matrices: values between -0.1 and 0.1 for stable weights
    uniform_real_distribution<float> dist_matrix(-0.1f, 0.1f);
    
    // For diagonal matrix d: values between 0.5 and 1.5 for scaling
    uniform_real_distribution<float> dist_diag(0.5f, 1.5f);

    // Initialize matrices with random values
    cout << "Initializing matrices with random values..." << endl;
    
    for (int i = 0; i < sizeA; i++) {
        A[i] = dist_matrix(gen);
    }
    
    for (int i = 0; i < sized; i++) {
        d[i] = dist_diag(gen);
    }
    
    for (int i = 0; i < sizeB; i++) {
        B[i] = dist_matrix(gen);
    }

    // Allocate memory on the device
    float* d_A;
    float* d_d;
    float* d_B;
    float* d_C;
    float* d_temp;
    
    cudaMalloc((void**)&d_A, sizeA * sizeof(float));
    cudaMalloc((void**)&d_d, sized * sizeof(float));
    cudaMalloc((void**)&d_B, sizeB * sizeof(float));
    cudaMalloc((void**)&d_C, sizeC * sizeof(float));
    cudaMalloc((void**)&d_temp, sizeTemp * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d, sized * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB * sizeof(float), cudaMemcpyHostToDevice);

    // Set up the execution configuration - using 16x16 block size
    dim3 blockDim(16, 16);
    dim3 gridDim1((colsA + blockDim.x - 1) / blockDim.x, 
                 (rowsA + blockDim.y - 1) / blockDim.y);
                 
    dim3 gridDim2((colsB + blockDim.x - 1) / blockDim.x, 
                 (rowsA + blockDim.y - 1) / blockDim.y);

    // CUDA events for timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Start timing
    cudaEventRecord(start);
    
    // First step: A * d -> temp
    multiplyWithDiagonal<<<gridDim1, blockDim>>>(d_A, d_d, d_temp, rowsA, colsA);
    
    // Second step: temp * B -> C
    matrixMultiply<<<gridDim2, blockDim>>>(d_temp, d_B, d_C, rowsA, colsA, colsB);

    // End timing
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    cout << "Execution time: " << milliseconds << " ms" << endl;

    // Copy the result back to host
    cudaMemcpy(C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    // Print sample of diagonal matrix d (first 5 elements or all if less than 5)
    cout << "Sample from diagonal matrix d: ";
    for (int i = 0; i < min(5, sized); i++) {
        cout << d[i] << " ";
    }
    cout << endl;

    // Print sample results from the final weight matrix C
    cout << "Sample from resulting weight matrix (A*d*B):" << endl;
    
    // Print first few elements
    cout << "First elements: ";
    for (int i = 0; i < min(5, sizeC); i++) {
        cout << C[i] << " ";
    }
    cout << endl;
    
    // Print some elements from the middle
    if (sizeC > 10) {
        int midIndex = sizeC / 2;
        cout << "Middle elements: ";
        for (int i = midIndex; i < min(midIndex + 5, sizeC); i++) {
            cout << C[i] << " ";
        }
        cout << endl;
    }
    
    // Print last few elements
    if (sizeC > 5) {
        cout << "Last elements: ";
        for (int i = max(0, sizeC - 5); i < sizeC; i++) {
            cout << C[i] << " ";
        }
        cout << endl;
    }

    // Free memory
    delete[] A;
    delete[] d;
    delete[] B;
    delete[] C;
    delete[] temp;
    
    cudaFree(d_A);
    cudaFree(d_d);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_temp);
    
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cout << "Weight vector for sparse fine-tuning successfully generated." << endl;
    return 0;
}
