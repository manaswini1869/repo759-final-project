#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <string>    // For std::stoi, std::string
#include <cmath>     // For fabs in comparison
#include <cufft.h>        // CUDA FFT library

static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorString(error);
}
void print_matrix(const float* data, int rows, int cols, int print_rows = 10, int print_cols = 10, const char* title = "Matrix") {
    std::cout << title << " (" << rows << "x" << cols << "):\n";
    // Limit printing for large matrices
    int r_limit = std::min(rows, print_rows);
    int c_limit = std::min(cols, print_cols);
    for (int i = 0; i < r_limit; ++i) {
        for (int j = 0; j < c_limit; ++j) {
            std::cout << std::fixed << std::setw(10) << std::setprecision(4)
                      << data[i * cols + j] << " ";
        }
        if (cols > c_limit) std::cout << "...";
        std::cout << "\n";
    }
     if (rows > r_limit) std::cout << "...\n";
}

// Function to print a complex matrix (host or managed memory accessible by host)
void print_complex_matrix(const cufftComplex* data, int rows, int cols, int print_rows = 10, int print_cols = 10, const char* title = "Complex Matrix") {
    std::cout << title << " (" << rows << "x" << cols << " complex elements):\n";
    // Limit printing
    int r_limit = std::min(rows, print_rows);
    int c_limit = std::min(cols, print_cols);
    for (int i = 0; i < r_limit; ++i) {
        for (int j = 0; j < c_limit; ++j) {
            std::cout << std::fixed << std::setw(10) << std::setprecision(4)
                      << "(" << data[i * cols + j].x << "," << data[i * cols + j].y << ") ";
        }
         if (cols > c_limit) std::cout << "...";
        std::cout << "\n";
    }
     if (rows > r_limit) std::cout << "...\n";
}

__global__ void complexElementwiseMultiply(const cufftComplex* a, const cufftComplex* b, cufftComplex* result, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a_re = a[idx].x;
        float a_im = a[idx].y;
        float b_re = b[idx].x;
        float b_im = b[idx].y;
        result[idx].x = a_re * b_re - a_im * b_im; // Real part
        result[idx].y = a_re * b_im + a_im * b_re; // Imaginary part
    }
}

__global__ void normalizeReal(float* data, size_t n, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale_factor;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <rows> <cols> <num_threads_per_block>\n";
        return;
    }

    int rows, cols, num_threads_per_block;

    rows = std::stoi(argv[1]);
    cols = std::stoi(argv[2]);
    num_threads_per_block = std::stoi(argv[3]);

    if (rows <= 0 || cols <= 0 || num_threads_per_block <= 0) {
        std::cerr << "Error: rows, cols, and num_threads_per_block must be positive integers.\n";
        return EXIT_FAILURE;
    }
    size_t size = static_cast<size_t>(rows) * static_cast<size_t>(cols);

    // --- Setup Random Number Generation ---
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    const float min_val = -1.0f, max_val = 1.0f;
    std::uniform_real_distribution<float> dist(min_val, max_val);

    // Managed memory is accessible from both CPU and GPU
    float *F = nullptr;         // Input real matrix
    cufftComplex *F_fft = nullptr; // Output of forward FFT (complex)
    cufftComplex *C = nullptr;         // Complex matrix for multiplication the noise stuff that should be multiplied (refer the fine-tuning pdf)
    cufftComplex *Mult_Result = nullptr; // Result of F_fft * C (complex)
    float *F_result = nullptr;  // Final result after inverse FFT (real)

    CHECK_CUDA(cudaMallocManaged((void **)&F, sizeof(float) * size));

    size_t complex_cols = static_cast<size_t>(cols) / 2 + 1;
    size_t complex_elements = static_cast<size_t>(rows) * complex_cols;

    cudaMallocManaged((void **)&F_fft, sizeof(cufftComplex) * complex_elements);
    cudaMallocManaged((void **)&C, sizeof(cufftComplex) * complex_elements);
    cudaMallocManaged((void **)&Mult_Result, sizeof(cufftComplex) * complex_elements);
    cudaMallocManaged((void **)&F_result, sizeof(float) * size);

    for (size_t i = 0; i < size; i++) {
        F[i] = dist(generator);
    }
    cudaDeviceSynchronize();

    // Initialize complex matrix C (e.g., random complex numbers or load from ckpt)
    // For this example, fill with random complex numbers
    // std::cout << "Initializing complex matrix C with random complex numbers...\n";
    // std::uniform_real_distribution<float> complex_dist(min_val, max_val);
    // for (size_t i = 0; i < complex_elements; ++i) {
    //     C[i].x = complex_dist(generator); // Real part
    //     C[i].y = complex_dist(generator); // Imaginary part
    // }
    // // Wait for host writes to be visible to the device
    // CHECK_CUDA(cudaDeviceSynchronize());
    //  // Optional: Print C (for small sizes)
    // if (rows <= 16 && complex_cols <= 16) {
    //     print_complex_matrix(C, rows, complex_cols, rows, complex_cols, "Complex Matrix C");
    // }


    // Setup cuFFT Plans
    cufftHandle plan_fwd; // Forward R2C
    cufftHandle plan_inv; // Inverse C2R
    cufftCreate(&plan_fwd);
    cufftPlan2d(&plan_fwd, rows, cols, CUFFT_R2C);

    cufftCreate(&plan_inv);
    cufftPlan2d(&plan_inv, rows, cols, CUFFT_C2R);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cufftExecR2C(plan_fwd, (cufftReal*)F, F_fft); // fourier transform cufftReal from ckpt probably ask harsha

    int block_size = num_threads_per_block;
    int grid_size = (complex_elements + block_size - 1) / block_size;
    complexElementwiseMultiply<<<grid_size, block_size>>>(F_fft, C, Mult_Result, complex_elements); // multiplication between F x C

    cufftExecC2R(plan_inv, Mult_Result, F_result); // inverse fourier transform on the resulting (FxC) or should it be (F x C x F`) F` = inverse transform on the initial F

    float scale_factor = 1.0f / static_cast<float>(size);
    grid_size = (size + block_size - 1) / block_size;
    normalizeReal<<<grid_size, block_size>>>(F_result, size, scale_factor);
    cudaGetLastError();

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop); /


    // --- Verify Result (Optional) ---
    // If C was designed such that F_result should ideally be equal to F (e.g., if C was all {1,0}),
    // you could compare F and F_result here.
     if (rows <= 16 && cols <= 16) {
        print_matrix(F_result, rows, cols, rows, cols, "Final Result Matrix F_result (Normalized)");

        // Simple comparison
        float max_diff = 0.0f;
        for(size_t i = 0; i < size; ++i) {
            max_diff = std::max(max_diff, std::fabs(F[i] - F_result[i]));
        }
        std::cout << "Max absolute difference between initial F and final F_result: "
                  << std::scientific << max_diff << std::endl;
     }


    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Destroy cuFFT plans
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);

    // Free memory
    cudaFree(F);
    cudaFree(F_fft);
    cudaFree(C);
    cudaFree(Mult_Result);
    cudaFree(F_result);
    return ;
}