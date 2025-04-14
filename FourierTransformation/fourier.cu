// nvcc fourier_finetune.cu -lcufft -o fourier_finetune
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <set>
#include <cufft.h>


void random_init(float* matrix, int size) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < size; ++i) {
        matrix[i] = dist(gen);
    }
}

void apply_frequency_delta(cufftComplex* freq, int rows, int cols, int C) {
    std::set<std::pair<int, int>> freq_positions;
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    while (freq_positions.size() < C) {
        int i = gen() % rows;
        int j = gen() % cols;
        freq_positions.insert({i, j});
    }

    for (const auto& pos : freq_positions) {
        int idx = pos.first * cols + pos.second;
        freq[idx].x += dist(gen);
        freq[idx].y += dist(gen);
    }
}

void print_matrix(const float* data, int rows, int cols, const char* title = "Matrix") {
    std::cout << title << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setw(8) << std::setprecision(2)
                      << data[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <rows> <cols> <C>\n";
        return EXIT_FAILURE;
    }

    int rows, cols, C;
    *rows = std::stoi(argv[1]);
    *cols = std::stoi(argv[2]);
    *C = std::stoi(argv[3]);
    if (*rows <= 0 || *cols <= 0 || *C <= 0) {
        std::cerr << "Error: rows, cols, and C must be given and should be positive\n";
        exit(EXIT_FAILURE);
    }
    int size = rows * cols;

    // Host memory
    std::vector<float> h_W(size);
    std::vector<float> h_W_prime(size);

    random_init(h_W.data(), size);

    float* d_W;
    float* d_W_prime;
    cufftComplex* d_freq;

    CHECK_CUDA(cudaMalloc(&d_W, sizeof(float) * size));
    CHECK_CUDA(cudaMalloc(&d_W_prime, sizeof(float) * size));
    CHECK_CUDA(cudaMalloc(&d_freq, sizeof(cufftComplex) * size));
    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), sizeof(float) * size, cudaMemcpyHostToDevice));

    cufftHandle plan_forward, plan_inverse;
    cufftPlan2d(&plan_forward, rows, cols, CUFFT_R2C);
    cufftPlan2d(&plan_inverse, rows, cols, CUFFT_C2R);

    cufftExecR2C(plan_forward, d_W, d_freq);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<cufftComplex> h_freq(size);
    CHECK_CUDA(cudaMemcpy(h_freq.data(), d_freq, sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost));
    apply_frequency_delta(h_freq.data(), rows, cols, C);
    CHECK_CUDA(cudaMemcpy(d_freq, h_freq.data(), sizeof(cufftComplex) * size, cudaMemcpyHostToDevice));

    // Inverse FFT: F⁻¹(F(W) + ΔF) = W'
    cufftExecC2R(plan_inverse, d_freq, d_W_prime);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back result
    CHECK_CUDA(cudaMemcpy(h_W_prime.data(), d_W_prime, sizeof(float) * size, cudaMemcpyDeviceToHost));

    for (auto& val : h_W_prime) {
        val /= size;
    }

    std::cout << "Frequency fine-tuning complete.\n";
    print_matrix(h_W_prime.data(), rows, cols, "W' (after ΔFourier)");

    // Cleanup
    cudaFree(d_W);
    cudaFree(d_W_prime);
    cudaFree(d_freq);
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);

    return 0;
}
