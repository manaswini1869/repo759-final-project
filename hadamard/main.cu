// ==========================================
// main.cu
// ==========================================
#include <iostream>
#include <vector>
#include <string>
#include "hadamard.h"
#include "matrix_ops.h"
#include "utils.h"
#include <cuda_runtime.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <x_input_filename.bin>\n";
        return 1;
    }

    std::string xfile = argv[1];

    // Create CUDA events for profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Step 1: Load C values and locations
    std::vector<float> C_values;
    std::vector<int> C_locs;
    int C_rows, C_cols;
    load_sparse_C("inputs_bin/C_values.bin", "inputs_bin/C_locs.bin", C_values, C_locs, C_rows, C_cols);

    // Step 2: Find next powers of 2
    int next_pow2_rows = next_power_of_2(C_rows);
    int next_pow2_cols = next_power_of_2(C_cols);

    // Step 3: Create Hadamard matrices
    float* d_H_row = nullptr;
    float* d_H_col = nullptr;
    create_hadamard_matrix(next_pow2_rows, &d_H_row);
    create_hadamard_matrix(next_pow2_cols, &d_H_col);

    // Start timing
    cudaEventRecord(start);

    // Step 4: Reconstruct and pad C matrix
    float* d_C = nullptr;
    reconstruct_C(C_values, C_locs, C_rows, C_cols, next_pow2_rows, next_pow2_cols, &d_C);
    
    // Ensure C is fully ready before using it
    cudaDeviceSynchronize();

    // Step 5: Calculate deltaW = H * C * H
    float* d_deltaW = nullptr;
    calculate_deltaW(d_H_row, d_C, d_H_col, next_pow2_rows, next_pow2_cols, &d_deltaW);

    // Ensure deltaW is fully ready
    cudaDeviceSynchronize();

    // Step 6: Compute Y = deltaW * X
    compute_Y(d_deltaW, xfile, next_pow2_cols, next_pow2_rows);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "\n All computations done on CUDA!\n";
    std::cout << " CUDA Execution Time: " << milliseconds << " ms\n";

    // Cleanup
    cudaFree(d_H_row);
    cudaFree(d_H_col);
    cudaFree(d_C);
    cudaFree(d_deltaW);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
