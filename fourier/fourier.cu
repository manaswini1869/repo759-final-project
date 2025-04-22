#include <iostream>
#include <cmath>
#include <vector>
#include <cufft.h>
#include "load_ckpt.h"
#include "cnpy.h"
#include <cuda_fp16.h>
#include <cuComplex.h>

__global__ void extract_real_normalized(float* real_out, cufftComplex* complex_in, int size, float norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    real_out[idx] = complex_in[idx].x * norm;
}

__global__ void batched_matvec_kernel(const float* mat, const float* vecs, float* outs,
                                      int batch_size, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (row >= N || b >= batch_size) return;

    float sum = 0.0f;
    const float* vec = vecs + b * M;
    for (int i = 0; i < M; ++i) {
        sum += vec[i] * mat[i * N + row];
    }
    outs[b * N + row] = sum;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " /input/input_x.npy [threads_per_block]\n";
        return 1;
    }

    std::string xfile = argv[1];
    int block_size = 256;
    if (argc == 3) {
        block_size = std::stoi(argv[2]);
        if (block_size <= 0 || block_size > 1024) {
            std::cerr << "Invalid block size. Must be between 1 and 1024.\n";
            return 1;
        }
    }

    int ct_mat_rows = 2304, ct_mat_cols = 1024;
    size_t mat_size = ct_mat_rows * ct_mat_cols;
    // step 1: creating the sparse vector
    std::string ct_directory = "./fourier-checkpoints/gemma-2-2b-fourier-CT.npy";
    float* ct_host = nullptr;
    auto [ct_rows, ct_cols] = load_ckpt_float(ct_directory, ct_host);

    std::string locs_directory = "./fourier-checkpoints/gemma-2-2b-fourier-locs.npy";
    int* locs_host = nullptr;
    auto [locs_rows, locs_cols] = load_ckpt_int(locs_directory, locs_host);

    cufftComplex* freq_domain;
    cudaMalloc(&freq_domain, mat_size * sizeof(cufftComplex));
    cudaMemset(freq_domain, 0, mat_size * sizeof(cufftComplex));

    for (int i = 0; i < locs_cols; ++i) {
        int u = locs_host[i];
        int v = locs_host[locs_cols + i];
        int idx = u * ct_mat_cols + v;
        cufftComplex val = {ct_host[i], 0.0f};
        cudaMemcpy(&freq_domain[idx], &val, sizeof(cufftComplex), cudaMemcpyHostToDevice);
    }

    cufftComplex* time_domain;
    cudaMalloc(&time_domain, mat_size * sizeof(cufftComplex));

    cufftHandle plan;
    cufftResult result = cufftPlan2d(&plan, ct_mat_rows, ct_mat_cols, CUFFT_C2C);
    if (result != CUFFT_SUCCESS) {
        std::cerr << "cuFFT plan creation failed with error code: " << result << "\n";
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    result = cufftExecC2C(plan, freq_domain, time_domain, CUFFT_INVERSE);
    if (result != CUFFT_SUCCESS) {
        std::cerr << "cuFFT execution failed with error code: " << result << "\n";
        return 1;
    }

    float* real_part;
    cudaMalloc(&real_part, mat_size * sizeof(float));
    int blocks = (mat_size + block_size - 1) / block_size;
    float norm_factor = 1.0f / static_cast<float>(mat_size);
    extract_real_normalized<<<blocks, block_size>>>(real_part, time_domain, mat_size, norm_factor);
    cudaDeviceSynchronize();

    cnpy::NpyArray x_np = cnpy::npy_load(xfile);
    size_t B = x_np.shape[0];
    size_t D = x_np.shape[1];

    if (D != ct_mat_rows) {
        std::cerr << "Input dimension mismatch. Expected " << ct_mat_rows << ", got " << D << "\n";
        return 1;
    }

    std::vector<float> X_host(B * D);
    const uint16_t* x_data = x_np.data<uint16_t>();
    for (size_t i = 0; i < B * D; ++i) {
        __half h_val;
        reinterpret_cast<uint16_t&>(h_val) = x_data[i];
        X_host[i] = __half2float(h_val);
    }

    float* d_X;
    cudaMalloc(&d_X, X_host.size() * sizeof(float));
    cudaMemcpy(d_X, X_host.data(), X_host.size() * sizeof(float), cudaMemcpyHostToDevice);

    float* d_Y;
    cudaMalloc(&d_Y, B * ct_mat_cols * sizeof(float));

    dim3 vecBlock(block_size);
    dim3 vecGrid((ct_mat_cols + block_size - 1) / block_size, B);
    batched_matvec_kernel<<<vecGrid, vecBlock>>>(real_part, d_X, d_Y, B, ct_mat_rows, ct_mat_cols);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "cuFFT inverse time: " << ms << " ms\n";

    delete[] ct_host;
    delete[] locs_host;
    cudaFree(freq_domain);
    cudaFree(time_domain);
    cudaFree(real_part);
    cudaFree(d_X);
    cudaFree(d_Y);
    cufftDestroy(plan);

    return 0;
}
