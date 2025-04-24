#include <iostream>
#include "load_ckpt.h"
#include "matmul.cuh"
#include "construct_frame.h"


int main(int argc, char* argv[]) {
    int ct_mat_rows = atoi(argv[1]);  // 2304;
    int ct_mat_cols = atoi(argv[2]);  //1024;
    int l_m = atoi(argv[3]);  // 2;
    int l_n = atoi(argv[4]);  // 2;
    int k_m = ct_mat_rows / l_m;
    int k_n = ct_mat_cols / l_n;
    int num_tokens = atoi(argv[5]);  // 16;
    int threads_per_block = atoi(argv[6]);  // 256;
    std::string ct_directory = argv[7];  // "/home/harsha/proj/ece759-final-proj/checkpoints/Gemma-2-2b/value/frame/Gemma-2-2b-frame-value-CT.npy";
    std::string locs_directory = argv[8];  // "/home/harsha/proj/ece759-final-proj/checkpoints/Gemma-2-2b/value/frame/Gemma-2-2b-frame-value-locs.npy";
    std::string x_directory = argv[9];  // "/home/harsha/proj/ece759-final-proj/checkpoints/Gemma-2-2b/inputs/x_16.npy";

    // load the coefficients
    float* ct = nullptr; 
    auto [ct_rows, ct_cols] = load_ckpt_float(ct_directory, ct);

    // load the locations
    int* locs = nullptr; 
    auto [locs_rows, locs_cols] = load_ckpt_int(locs_directory, locs);

    // load the tokens
    float* x = nullptr; 
    auto [x_rows, x_cols] = load_ckpt_float(x_directory, x);

    float* tff_m = nullptr;
    float* tff_n = nullptr;

    tff_m = construct_real_tff(k_m, l_m/2, ct_mat_rows/2);
    tff_n = construct_real_tff(k_n, l_n/2, ct_mat_cols/2);

    // multiply tff_m * ct_mat * tff_n.T in cuda
    float *d_tff_m, *d_tff_n, *d_ct, *d_D1, *d_y, *d_x;
    float *d_tff_m_unfold, *d_tff_n_unfold;
    int *d_locs;

    // Allocate memory
    cudaMalloc(&d_tff_m, ct_mat_rows * ct_mat_rows * sizeof(float));
    cudaMalloc(&d_tff_n, ct_mat_cols * ct_mat_cols * sizeof(float));
    cudaMalloc(&d_tff_m_unfold, ct_mat_rows * ct_cols * sizeof(float));
    cudaMalloc(&d_tff_n_unfold, ct_cols * ct_mat_cols * sizeof(float));

    cudaMalloc(&d_ct, ct_cols * sizeof(float));
    cudaMalloc(&d_locs, locs_rows * locs_cols * sizeof(int));

    cudaMalloc(&d_y, num_tokens * ct_mat_cols * sizeof(float));
    cudaMalloc(&d_x, num_tokens * ct_mat_rows * sizeof(float));

    cudaMalloc(&d_D1, num_tokens * ct_cols * sizeof(float));

    // Copy data
    cudaMemcpy(d_tff_m, tff_m, ct_mat_rows * ct_mat_rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tff_n, tff_n, ct_mat_cols * ct_mat_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ct, ct, ct_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_locs, locs, locs_rows * locs_cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, num_tokens * ct_mat_rows * sizeof(float), cudaMemcpyHostToDevice);

    // Timing with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    frame_compute_y_3(  d_tff_m,
                        d_tff_n,
                        d_ct,
                        d_locs,
                        d_tff_m_unfold,
                        d_tff_n_unfold,
                        d_x,
                        d_D1,
                        d_y,
                        num_tokens,
                        ct_mat_rows,
                        ct_mat_cols,
                        ct_cols,
                        threads_per_block);
    cudaEventRecord(stop);

    // Wait for kernel to finish
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // // Copy result back to host
    // int bytes = num_tokens * ct_mat_cols * sizeof(float);
    // float* h_y = new float[num_tokens * ct_mat_cols];
    // cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

    std::cout << milliseconds << std::endl;

    // for DEBUG
    // save_array("/home/harsha/proj/ece759-final-proj/temp.npy", ct_mat, ct_mat_rows * ct_mat_cols);

    delete[] ct;
    delete[] locs;
    delete[] tff_m;
    delete[] tff_n;
    // delete[] h_y;
    cudaFree(d_tff_m);
    cudaFree(d_tff_n);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}