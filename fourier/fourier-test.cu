#include <iostream>
#include <cmath>
#include <vector>
#include "load_ckpt.h"
#include "cnpy.h"
#include <cuda_fp16.h>

#define PI 3.14159265358979323846

__global__ void sparse_ifft2_kernel(float* real_out, float* imag_out,
                                    const float* ct, const int* locs,
                                    int n, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= M || y >= N) return;

    float real_sum = 0.0f;
    float imag_sum = 0.0f;

    for (int i = 0; i < n; ++i) {
        int u = locs[i];
        int v = locs[n + i];
        float coeff = ct[i];
        float angle = 2.0f * PI * (u * x / (float)M + v * y / (float)N);
        real_sum += coeff * cosf(angle);
        imag_sum += coeff * sinf(angle);
    }

    int index = x * N + y;
    real_out[index] = real_sum;
    imag_out[index] = imag_sum;
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
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " /input/input_x.npy\n";
        return 1;
    }

    std::string xfile = argv[1];

    int ct_mat_rows = 2304, ct_mat_cols = 1024;
    size_t mat_size = ct_mat_rows * ct_mat_cols;

    std::string ct_directory = "./fourier-checkpoints/gemma-2-2b-fourier-CT.npy";
    float* ct_host = nullptr;
    auto [ct_rows, ct_cols] = load_ckpt_float(ct_directory, ct_host);

    std::string locs_directory = "./fourier-checkpoints/gemma-2-2b-fourier-locs.npy";
    int* locs_host = nullptr;
    auto [locs_rows, locs_cols] = load_ckpt_int(locs_directory, locs_host);

    float *ct_device, *real_device, *imag_device;
    int* locs_device;
    cudaMalloc(&ct_device, locs_cols * sizeof(float));
    cudaMalloc(&locs_device, 2 * locs_cols * sizeof(int));
    cudaMalloc(&real_device, mat_size * sizeof(float));
    cudaMalloc(&imag_device, mat_size * sizeof(float));
    cudaMemcpy(ct_device, ct_host, locs_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(locs_device, locs_host, 2 * locs_cols * sizeof(int), cudaMemcpyHostToDevice);

    // Load and convert input matrix (shape: B x 2304)
    cnpy::NpyArray x_np = cnpy::npy_load(xfile);
    size_t B = x_np.shape[0];
    size_t D = x_np.shape[1];

    if (D != ct_mat_rows) {
        std::cerr << "Input dimension mismatch. Expected " << ct_mat_rows
                  << ", got " << D << "\n";
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

    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 72, 256, 1024};

    std::cout << "Benchmarking sparse inverse FFT kernel with batch size " << B << ":\n";

    for (int total_threads : thread_counts) {
        int block_x = (int)sqrt(total_threads);
        int block_y = total_threads / block_x;
        while (block_x * block_y != total_threads && block_x > 1) {
            block_x--;
            block_y = total_threads / block_x;
        }

        if (block_x * block_y != total_threads) {
            std::cout << "Skipping " << total_threads << " threads — cannot form valid 2D block\n";
            continue;
        }

        dim3 threadsPerBlock(block_x, block_y);
        dim3 numBlocks((ct_mat_cols + block_x - 1) / block_x,
                       (ct_mat_rows + block_y - 1) / block_y);

        cudaMemset(real_device, 0, mat_size * sizeof(float));
        cudaMemset(imag_device, 0, mat_size * sizeof(float));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        sparse_ifft2_kernel<<<numBlocks, threadsPerBlock>>>(
            real_device, imag_device, ct_device, locs_device,
            locs_cols, ct_mat_rows, ct_mat_cols);

        // Batched matrix-vector multiplication
        dim3 vecBlock(256);
        dim3 vecGrid((ct_mat_cols + vecBlock.x - 1) / vecBlock.x, B);
        batched_matvec_kernel<<<vecGrid, vecBlock>>>(real_device, d_X, d_Y, B, ct_mat_rows, ct_mat_cols);

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        std::cout << "Threads: " << total_threads
                  << " (Block: " << block_x << "x" << block_y << ")"
                  << " → Time: " << ms << " ms\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Cleanup
    delete[] ct_host;
    delete[] locs_host;
    cudaFree(ct_device);
    cudaFree(locs_device);
    cudaFree(real_device);
    cudaFree(imag_device);
    cudaFree(d_X);
    cudaFree(d_Y);

    return 0;
}
