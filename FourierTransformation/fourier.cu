// nvcc fourierft_cuda.cu -lcufft -o fourierft_cuda
#include <iostream>
#include <vector>
#include <random>
#include <set>
#include <cufft.h>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <rows> <cols> <C>" << std::endl;
        return 1;
    }

    int rows = std::stoi(argv[1]);
    int cols = std::stoi(argv[2]);
    int C = std::stoi(argv[3]);
    int size = rows * cols;

    // aandom frequency selection
    std::set<std::pair<int, int>> freq_positions;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    while (freq_positions.size() < C) {
        int i = gen() % rows;
        int j = gen() % cols;
        freq_positions.insert({i, j});
    }

    // create host-side frequency domain matrix (complex)
    cufftComplex* h_freq = new cufftComplex[size];
    for (int i = 0; i < size; ++i) {
        h_freq[i].x = 0.0f;
        h_freq[i].y = 0.0f;
    }

    // set spectral coefficients
    for (const auto& pos : freq_positions) {
        int idx = pos.first * cols + pos.second;
        h_freq[idx].x = dist(gen); // real
        h_freq[idx].y = dist(gen); // imag
    }

    // sllocate device memory
    cufftComplex* d_freq;
    float* d_spatial;
    CHECK_CUDA(cudaMalloc(&d_freq, sizeof(cufftComplex) * size));
    CHECK_CUDA(cudaMalloc(&d_spatial, sizeof(float) * size));
    CHECK_CUDA(cudaMemcpy(d_freq, h_freq, sizeof(cufftComplex) * size, cudaMemcpyHostToDevice));

    // setup cuFFT plan
    cufftHandle plan;
    if (cufftPlan2d(&plan, rows, cols, CUFFT_C2R) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT plan creation failed!" << std::endl;
        return 1;
    }

    // execute inverse FFT
    if (cufftExecC2R(plan, d_freq, d_spatial) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT execution failed!" << std::endl;
        return 1;
    }

    // copy result back to host
    std::vector<float> h_spatial(size);
    CHECK_CUDA(cudaMemcpy(h_spatial.data(), d_spatial, sizeof(float) * size, cudaMemcpyDeviceToHost));

    // normalize and print result
    std::cout << "Reconstructed ΔW (spatial domain):" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float val = h_spatial[i * cols + j] / size;
            std::cout << std::fixed << std::setw(8) << std::setprecision(2) << val << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    delete[] h_freq;
    cufftDestroy(plan);
    cudaFree(d_freq);
    cudaFree(d_spatial);

    return 0;
}
