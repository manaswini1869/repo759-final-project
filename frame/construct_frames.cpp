#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "construct_frame.h"


int tffet(int k, int l, int n)
{
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


void insert_tx(float x, float* mat, int rows, int cols, int r, int c)
{
    float a = sqrtf(x);
    float b = sqrtf(2.0f - x);
    float f = 1.0f / sqrtf(2.0f);

    mat[r * cols + c]         = f * a;
    mat[r * cols + (c + 1)]   = f * a;
    mat[(r + 1) * cols + c]   = f * b;
    mat[(r + 1) * cols + (c + 1)] = -f * b;
}


Complex* construct_tight_frames(int k, int l, int n) 
{
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

float* construct_real_tff(int k, int l, int n)
{
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