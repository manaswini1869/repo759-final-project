#include <iostream>
#include "load_ckpt.h"
#include "matmul.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct {
    float real;
    float imag;
} Complex;


int tffet(int k, int l, int n) {
    if (2 * l > n)
        l = n - l;

    int exists = -1;

    while (exists == -1) {
        if (n % l == 0) {
            exists = (k >= n / l);
        } else {
            int ceil_div = (int)ceil((float)n / l);
            if (k > ceil_div + 1) exists = 1;
            else if (k < ceil_div + 1) exists = 0;
            else {
                n = k * l - n;
                l = n - l;
            }
        }
    }

    return exists;
}


void insert_tx(float x, float* mat, int rows, int cols, int r, int c) {
    float a = sqrtf(x);
    float b = sqrtf(2.0f - x);
    float f = 1.0f / sqrtf(2.0f);

    mat[r * cols + c]         = f * a;
    mat[r * cols + (c + 1)]   = f * a;
    mat[(r + 1) * cols + c]   = f * b;
    mat[(r + 1) * cols + (c + 1)] = -f * b;
}


Complex* construct_tight_frames(int k, int l, int n) {
    if (!tffet(k, l, n)) {
        printf("Invalid k, l, n values\n");
        printf("k: %d, l: %d, n: %d\n", k, l, n);
        exit(1);
    }

    float* frame = (float*)calloc(l * n, sizeof(float));
    Complex* tffs = (Complex*)malloc(k * l * n * sizeof(Complex));

    float target_norm = (float)n / l;
    int col = 0;

    for (int row = 0; row < l; ++row) {
        float curr_norm = 0;
        for (int j = 0; j < col; ++j)
            curr_norm += frame[row * n + j] * frame[row * n + j];

        float req_norm = target_norm - curr_norm;

        while (req_norm >= 1.0f || fabsf(req_norm - 1.0f) < 1e-5) {
            frame[row * n + col] = 1.0f;
            req_norm -= 1.0f;
            col += 1;
        }

        if (fabsf(req_norm) > 1e-5) {
            insert_tx(req_norm, frame, l, n, row, col);
            col += 2;
        }
    }

    for (int _k = 0; _k < k; ++_k) {
        for (int row = 0; row < l; ++row) {
            for (int col = 0; col < n; ++col) {
                float theta = 2.0f * M_PI * _k * col / k;
                float re = cosf(theta);
                float im = sinf(theta);
                float val = frame[row * n + col];

                int idx = _k * l * n + row * n + col;
                tffs[idx].real = val * re;
                tffs[idx].imag = val * im;
            }
        }
    }

    free(frame);
    return tffs;
}

float* construct_real_tff(int k, int l, int n) {
    Complex* tffs = construct_tight_frames(k, l, n);

    int total_rows = 2 * l;
    int total_cols = 2 * n;
    float* out = (float*)calloc(k * total_rows * total_cols, sizeof(float));

    for (int _k = 0; _k < k; ++_k) {
        for (int row = 0; row < l; ++row) {
            for (int col = 0; col < n; ++col) {
                Complex z = tffs[_k * l * n + row * n + col];

                // Even
                int even_row = row;
                int even_col1 = 2 * col;
                int even_col2 = 2 * col + 1;

                int idx1 = _k * total_rows * total_cols + even_row * total_cols + even_col1;
                int idx2 = _k * total_rows * total_cols + even_row * total_cols + even_col2;

                out[idx1] = z.real * powf(-1, even_col1);
                out[idx2] = z.imag * powf(-1, even_col2);

                // Odd
                int odd_row = row + l;
                int odd_col1 = (2 * col + 1) % (2 * n);
                int odd_col2 = (2 * col + 2) % (2 * n);

                int idx3 = _k * total_rows * total_cols + odd_row * total_cols + odd_col1;
                int idx4 = _k * total_rows * total_cols + odd_row * total_cols + odd_col2;

                out[idx3] = z.real;
                out[idx4] = z.imag;
            }
        }
    }

    free(tffs);
    return out;
}


int main() {
    int ct_mat_rows = 2304, ct_mat_cols = 1024;
    int k_m = 576, l_m = 4, k_n = 256, l_n = 4;
    int num_tokens = 16;
    int threads_per_block = 256;

    float* h_y = new float[num_tokens * ct_mat_cols];
    float* h_D = new float[ct_mat_rows * ct_mat_cols];
    float* h_dw = new float[ct_mat_rows * ct_mat_cols];

    // load the coefficients
    std::string ct_directory = "/home/harsha/proj/ece759-final-proj/checkpoints/gemma-2-2b/frame/gemma-2-2b-frame-CT.npy";
    float* ct = nullptr; 
    auto [ct_rows, ct_cols] = load_ckpt_float(ct_directory, ct);

    // load the locations
    std::string locs_directory = "/home/harsha/proj/ece759-final-proj/checkpoints/gemma-2-2b/frame/gemma-2-2b-frame-locs.npy";
    int* locs = nullptr; 
    auto [locs_rows, locs_cols] = load_ckpt_int(locs_directory, locs);

    // load the tokens
    std::string x_directory = "/home/harsha/proj/ece759-final-proj/checkpoints/gemma-2-2b/inputs/x_16.npy";
    float* x = nullptr; 
    auto [x_rows, x_cols] = load_ckpt_float(x_directory, x);

    // create a float array of zeros with size ct_mat_rows * ct_mat_cols
    float* ct_mat = new float[ct_mat_rows * ct_mat_cols];
    for (size_t i = 0; i < ct_mat_rows * ct_mat_cols; ++i) {
        ct_mat[i] = 0.0f;
    }
    // copy the values of ct into the ct_mat at locations specified by locs
    for (size_t i = 0; i < locs_cols; ++i) {
        int coeff_loc = locs[i]*ct_mat_cols + locs[locs_cols + i];
        ct_mat[coeff_loc] = ct[i];
    }


    float* tff_m = nullptr;
    float* tff_n = nullptr;

    tff_m = construct_real_tff(k_m, l_m/2, ct_mat_rows/2);
    tff_n = construct_real_tff(k_n, l_n/2, ct_mat_cols/2);

    // multiply tff_m * ct_mat * tff_n.T in cuda
    float *d_tff_m, *d_tff_n, *d_ct_mat, *d_D, *d_dw, *d_y, *d_x;

    // Allocate memory
    cudaMalloc(&d_tff_m, ct_mat_rows * ct_mat_rows * sizeof(float));
    cudaMalloc(&d_tff_n, ct_mat_cols * ct_mat_cols * sizeof(float));
    cudaMalloc(&d_ct_mat, ct_mat_rows * ct_mat_cols * sizeof(float));
    cudaMalloc(&d_D, ct_mat_rows * ct_mat_cols * sizeof(float));
    cudaMalloc(&d_dw, ct_mat_rows * ct_mat_cols * sizeof(float));
    cudaMalloc(&d_y, num_tokens * ct_mat_cols * sizeof(float));
    cudaMalloc(&d_x, num_tokens * ct_mat_rows * sizeof(float));

    // Copy data
    cudaMemcpy(d_tff_m, tff_m, ct_mat_rows * ct_mat_rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tff_n, tff_n, ct_mat_cols * ct_mat_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ct_mat, ct_mat, ct_mat_rows * ct_mat_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, num_tokens * ct_mat_rows * sizeof(float), cudaMemcpyHostToDevice);

    // Timing with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    frame_compute_dw(d_tff_m, d_tff_n, d_ct_mat, d_D, d_dw, ct_mat_rows, ct_mat_cols, threads_per_block);

    frame_compute_y(d_x, d_dw, d_y, num_tokens, ct_mat_rows, ct_mat_cols, threads_per_block);
    cudaEventRecord(stop);

    // Wait for kernel to finish
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    int bytes = num_tokens * ct_mat_cols * sizeof(float);
    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    bytes = ct_mat_rows * ct_mat_cols * sizeof(float);
    cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost);
    bytes = ct_mat_rows * ct_mat_cols * sizeof(float);
    cudaMemcpy(h_dw, d_dw, bytes, cudaMemcpyDeviceToHost);

    // Print the last element
    std::cout << h_y[num_tokens * ct_mat_cols - 1] << std::endl;
    std::cout << h_D[ct_mat_rows * ct_mat_cols - 1] << std::endl;
    std::cout << h_dw[ct_mat_rows * ct_mat_cols - 1] << std::endl;
    std::cout << milliseconds << std::endl;

    // for DEBUG
    // save_array("/home/harsha/proj/ece759-final-proj/temp.npy", ct_mat, ct_mat_rows * ct_mat_cols);

    /*
    */

    delete[] ct;
    delete[] locs;
    delete[] ct_mat;
    delete[] tff_m;
    delete[] tff_n;
    delete[] h_y;
    delete[] h_D;
    delete[] h_dw;
    cudaFree(d_tff_m);
    cudaFree(d_tff_n);
    cudaFree(d_ct_mat);
    cudaFree(d_D);
    cudaFree(d_dw);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}