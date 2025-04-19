// ==========================================
// matrix_ops.h
// ==========================================
#pragma once

#include <vector>
#include <string>

void reconstruct_C(const std::vector<float>& values, const std::vector<int>& locs,
                   int rows, int cols, int padded_rows, int padded_cols, float** d_C);

void calculate_deltaW(float* d_H_row, float* d_C, float* d_H_col,
                      int rows, int cols, float** d_deltaW);

void compute_Y(float* d_deltaW, const std::string& xfile, int input_dim, int output_dim);

int next_power_of_2(int n);