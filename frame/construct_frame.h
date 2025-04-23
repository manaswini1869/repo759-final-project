#ifndef CONSTRUCT_FRAME_H
#define CONSTRUCT_FRAME_H

typedef struct {
    float real;
    float imag;
} Complex;

int tffet(int k, int l, int n);

void insert_tx(float x, float* mat, int rows, int cols, int r, int c);

Complex* construct_tight_frames(int k, int l, int n);

float* construct_real_tff(int k, int l, int n);

#endif