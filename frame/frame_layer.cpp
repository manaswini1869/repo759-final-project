#include <iostream>
#include "load_ckpt.h"


int main() {
    int ct_mat_rows = 2304, ct_mat_cols = 1024;

    // load the coefficients
    std::string ct_directory = "/home/harsha/proj/ece759-final-proj/checkpoints/gemma-2-2b/frame/gemma-2-2b-frame-CT.npy";
    float* ct = nullptr; // new float[num_nz_coeffs];
    auto [ct_rows, ct_cols] = load_ckpt_float(ct_directory, ct);

    // load the locations
    std::string locs_directory = "/home/harsha/proj/ece759-final-proj/checkpoints/gemma-2-2b/frame/gemma-2-2b-frame-locs.npy";
    int* locs = nullptr; // new int[num_nz_coeffs * 2];
    auto [locs_rows, locs_cols] = load_ckpt_int(locs_directory, locs);

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

    // for DEBUG
    // save_array("/home/harsha/proj/ece759-final-proj/temp.npy", ct_mat, ct_mat_rows * ct_mat_cols);

    delete[] ct;
    delete[] locs;

    return 0;
}