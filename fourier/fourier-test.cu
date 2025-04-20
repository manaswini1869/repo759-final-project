#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits> // Needed for numeric_limits

// CUDA specific includes
#include <cuda_runtime.h> // For cudaMallocManaged, cudaMemset, cudaFree, cudaDeviceSynchronize, etc.
// #include <cufft.h> // REMOVED

#include "load_ckpt.cuh" // Assuming this header is independent of cufft

// --- Custom FFT Implementation ---

// 1. Define Complex Number Struct
struct Complex {
    float x; // Real part
    float y; // Imaginary part

    // Default constructor
    __device__ __host__ Complex(float r = 0.0f, float i = 0.0f) : x(r), y(i) {}

    // Basic complex arithmetic (Device functions)
    __device__ __forceinline__ Complex operator+(const Complex& other) const {
        return Complex(x + other.x, y + other.y);
    }
    __device__ __forceinline__ Complex operator-(const Complex& other) const {
        return Complex(x - other.x, y - other.y);
    }
    __device__ __forceinline__ Complex operator*(const Complex& other) const {
        return Complex(x * other.x - y * other.y, x * other.y + y * other.x);
    }
     __device__ __forceinline__ Complex operator*(float scalar) const {
        return Complex(x * scalar, y * scalar);
    }
     // Conjugate helper
    __device__ __forceinline__ Complex conj() const {
        return Complex(x, -y);
    }
};

// Helper to check if a number is a power of 2
bool isPowerOfTwo(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

// Helper to get the next power of 2
int nextPowerOfTwo(int n) {
    if (n <= 0) return 1;
    if (isPowerOfTwo(n)) return n;
    int power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

// --- CUDA Kernels ---

// Kernel to copy real data into the real part of complex data, setting imaginary to zero
// Modified to use custom Complex struct
__global__ void floatToComplexRealOnly(const float *real_in,
                                     Complex *complex_out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        complex_out[idx].x = real_in[idx];
        complex_out[idx].y = 0.0f;
    }
}

// Kernel to copy real data into the real part of complex data with padding
__global__ void floatToComplexRealOnlyPadded(const float *real_in, Complex *complex_out_padded,
                                           int rows, int cols, int padded_rows, int padded_cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols) {
        size_t in_idx = static_cast<size_t>(r) * cols + c;
        size_t out_idx = static_cast<size_t>(r) * padded_cols + c; // Map to padded grid
        complex_out_padded[out_idx].x = real_in[in_idx];
        complex_out_padded[out_idx].y = 0.0f;
    }
    // Note: The rest of complex_out_padded should be zero-initialized beforehand
}


// --- Simplified 1D IFFT Kernel (Radix-2, In-place, requires power-of-2 size N) ---
// --- Based on typical Cooley-Tukey implementation structure        ---
// --- WARNING: This is a basic version, not optimized for performance ---
// --- Needs shared memory for actual speedup, error handling, etc. ---
__device__ unsigned int reverseBits(unsigned int n, unsigned int numBits) {
    unsigned int reversedN = 0;
    for (unsigned int i = 0; i < numBits; ++i) {
        if ((n >> i) & 1) {
            reversedN |= 1 << (numBits - 1 - i);
        }
    }
    return reversedN;
}


__global__ void ifft1D_kernel(Complex* data, int N, int numBits) {
    // This kernel processes ONE entire 1D array of size N.
    // Launch this kernel with 1 block and N threads, or adapt for multiple arrays.
    // Assumes N is power of 2.

    extern __shared__ Complex s_data[]; // Allocate N*sizeof(Complex) in shared memory
    unsigned int tid = threadIdx.x;
    // In this kernel launch configuration, blockDim.x == N
    // unsigned int block_size = blockDim.x; // Not strictly needed if blockDim.x == N

    // --- Explicitly zero shared memory ---
    // Ensure all threads relevant to the N-sized transform
    // clear their portion of the shared memory buffer.
    if (tid < N) {
        s_data[tid].x = 0.0f;
        s_data[tid].y = 0.0f;
    }
    // Synchronize to ensure all threads have finished zeroing before loading
    __syncthreads();

    // --- Load data into shared memory with bit reversal ---
    // We only load data from global memory for valid indices.
    // The positions in shared memory corresponding to invalid
    // bit-reversed indices will remain zeroed from the step above.
    if (tid < N) {
        unsigned int reversed_idx = reverseBits(tid, numBits);

        // Ensure the reversed index is within the bounds of the input data size N
        if (reversed_idx < N) {
             s_data[tid] = data[reversed_idx]; // Read data from global memory
        }
        // If reversed_idx >= N, the 'else' block is skipped, and s_data[tid]
        // remains the zero value set during the initial zeroing phase.
    }
    // Synchronize after loading to ensure all data is available in shared memory
    // before the butterfly computations begin.
    __syncthreads();

    // --- Butterfly Passes (Cooley-Tukey Radix-2 In-place) ---
     for (unsigned int s = 1; s <= numBits; ++s) {
         unsigned int m = 1 << s;      // Current size of sub-problems
         unsigned int m_half = m >> 1; // Half size

         // Twiddle factor angle base for this stage (IFFT: positive exponent)
         float angle_base = 2.0f * M_PI / m;

         // Parallel computation of butterflies within the stage
         // Each thread works on one pair of elements (idx1, idx2) if tid < N
         if (tid < N) {
              // 'k' is the index within the current sub-problem [0 to m_half - 1]
              // It also determines the twiddle factor W_m^k
              unsigned int k = tid % m_half;

              // Calculate the twiddle factor W_m^k = cos(angle) + j*sin(angle)
              float angle = angle_base * k;
              Complex w = Complex(__cosf(angle), __sinf(angle)); // IFFT twiddle factor

              // Find the indices for the butterfly operation
              // Start index of the group of 'm' elements containing tid
              unsigned int group_start = (tid / m_half) * m_half;
              // Index of the first element in the butterfly pair
              unsigned int idx1 = group_start + k;
              // Index of the second element in the butterfly pair
              unsigned int idx2 = idx1 + m_half;

              // Read elements from shared memory
              Complex a = s_data[idx1];
              Complex b = s_data[idx2] * w; // Multiply second element by twiddle factor

              // Perform the butterfly operation and write results back to shared memory
              s_data[idx1] = a + b;
              s_data[idx2] = a - b;
          }
          // Synchronize threads within the block after each stage of butterflies
          __syncthreads();
     }

     // --- Write result back to global memory ---
     // The scaling factor (1/N) is applied after the 2D FFT (in the main function)
     if (tid < N) {
         data[tid] = s_data[tid];
     }
}


// --- Kernel for Matrix Transposition (Tiled for better coalescing) ---
#define TILE_DIM 16 // Example tile size, can be tuned (e.g., 32)
__global__ void transposeComplex(const Complex* input, Complex* output, int width, int height) {
    __shared__ Complex tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x; // Global col index in input
    int y = blockIdx.y * TILE_DIM + threadIdx.y; // Global row index in input

    // Load tile from global input memory into shared memory
    if (x < width && y < height) {
        const Complex* src = input + static_cast<size_t>(y) * width + x;
	Complex* dst = &tile[threadIdx.y][threadIdx.x];
	memcpy(dst, src, sizeof(Complex));
    }

    __syncthreads();

    // Transpose within the block
    x = blockIdx.y * TILE_DIM + threadIdx.x; // Global col index in output (original row block)
    y = blockIdx.x * TILE_DIM + threadIdx.y; // Global row index in output (original col block)

    // Write transposed tile from shared memory to global output memory
    if (x < height && y < width) { // Note swapped width/height for output bounds
        output[static_cast<size_t>(y) * height + x] = tile[threadIdx.x][threadIdx.y]; // Transposed access
    }
}

// --- Kernel to extract real part and scale ---
__global__ void extractRealAndScale(const Complex *complex_in, float *real_out, size_t n, float scale_factor) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        real_out[idx] = complex_in[idx].x * scale_factor;
    }
}

// Kernel to extract real part and scale from PADDED data
__global__ void extractRealAndScalePadded(const Complex *complex_in_padded, float *real_out,
                                        int rows, int cols, int padded_cols, float scale_factor) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols) {
        size_t in_idx = static_cast<size_t>(r) * padded_cols + c; // Read from padded grid
        size_t out_idx = static_cast<size_t>(r) * cols + c;       // Write to original grid
        real_out[out_idx] = complex_in_padded[in_idx].x * scale_factor;
    }
}


// --- Utility Functions (Keep original print functions, but update complex one) ---

void print_matrix(const float *data, int rows, int cols, int print_rows = 10,
                  int print_cols = 10, const char *title = "Matrix") {
    std::cout << title << " (" << rows << "x" << cols << "):\n";
    int r_limit = std::min(rows, print_rows);
    int c_limit = std::min(cols, print_cols);
    if (!data) {
        std::cout << "(null pointer)\n";
        return;
    }
    cudaDeviceSynchronize(); // Ensure data is visible

    for (int i = 0; i < r_limit; ++i) {
        for (int j = 0; j < c_limit; ++j) {
            std::cout << std::fixed << std::setw(12) << std::setprecision(6)
                      << data[static_cast<size_t>(i) * cols + j] << " ";
        }
        if (cols > c_limit)
            std::cout << "...";
        std::cout << "\n";
    }
    if (rows > r_limit)
        std::cout << "...\n";
}

// Modified to use custom Complex struct
void print_complex_matrix(const Complex *data, int rows, int cols,
                          int print_rows = 10, int print_cols = 10,
                          const char *title = "Complex Matrix") {
    std::cout << title << " (" << rows << "x" << cols << " complex elements):\n";
    int r_limit = std::min(rows, print_rows);
    int c_limit = std::min(cols, print_cols);
    if (!data) {
        std::cout << " (null pointer)\n";
        return;
    }
    cudaDeviceSynchronize(); // Ensure data is visible

    for (int i = 0; i < r_limit; ++i) {
        for (int j = 0; j < c_limit; ++j) {
            std::cout << std::fixed << std::setw(18) << std::setprecision(6) << "("
                      << data[static_cast<size_t>(i) * cols + j].x << ","
                      << data[static_cast<size_t>(i) * cols + j].y << ") ";
        }
         if (cols > c_limit) std::cout << "...";
        std::cout << "\n";
    }
     if (rows > r_limit) std::cout << "...\n";
}

// --- Main Function (Modified) ---

int main(int argc, char *argv[]) {
    int ct_mat_rows = 2304;
    int ct_mat_cols = 1024; // Power of 2
    size_t ct_mat_size = static_cast<size_t>(ct_mat_rows) * ct_mat_cols;
    size_t ct_mat_bytes = ct_mat_size * sizeof(float);

    // Check if dimensions are power of 2 (for this simple FFT implementation)
    // Need padding for rows (2304)
    bool needs_padding_rows = !isPowerOfTwo(ct_mat_rows);
    bool needs_padding_cols = !isPowerOfTwo(ct_mat_cols); // 1024 is power of 2

    int padded_rows = needs_padding_rows ? nextPowerOfTwo(ct_mat_rows) : ct_mat_rows; // 4096
    int padded_cols = needs_padding_cols ? nextPowerOfTwo(ct_mat_cols) : ct_mat_cols; // 1024

    std::cout << "Original dimensions: " << ct_mat_rows << "x" << ct_mat_cols << std::endl;
    if (needs_padding_rows || needs_padding_cols) {
        std::cout << "Padding needed for custom FFT. Padded dimensions: "
                  << padded_rows << "x" << padded_cols << std::endl;
    }

    size_t padded_mat_size = static_cast<size_t>(padded_rows) * padded_cols;
    size_t padded_complex_bytes = padded_mat_size * sizeof(Complex);


    std::string ct_directory = "../checkpoints/gemma-2-2b/fourier/gemma-2-2b-fourier-CT.npy";
    std::string locs_directory = "../checkpoints/gemma-2-2b/fourier/gemma-2-2b-fourier-locs.npy";

    float *h_ct = nullptr;
    int *h_locs = nullptr;
    float *d_ct = nullptr;
    int *d_locs = nullptr;
    float *d_ct_mat = nullptr; // Original reconstructed C matrix
    Complex *d_freq_complex_padded = nullptr; // Padded complex data for FFT
    Complex *d_temp_transpose = nullptr;    // Temporary buffer for transpose
    float *d_spatial_real = nullptr;      // Final real output

    int ct_rows_loaded = 0, ct_cols_loaded = 0;
    int locs_rows_loaded = 0, locs_cols_loaded = 0;
    size_t ct_sparse_size = 0;
    size_t locs_sparse_size = 0;

    // Removed cufftHandle and plan_created

    try {
        // Steps 1-9: Load sparse data, reconstruct d_ct_mat (Same as original)
        // ... (Keep the original code for steps 1-9) ...
        // Assume d_ct_mat (ct_mat_rows x ct_mat_cols) is correctly reconstructed in managed memory
        // ... (Load ct, locs, allocate d_ct, d_locs, d_ct_mat, reconstruct d_ct_mat) ...

         std::cout << "Loading coefficients from " << ct_directory
              << " into host memory..." << std::endl;
        auto ct_dims = load_ckpt_host_float(ct_directory, h_ct);
        ct_rows_loaded = std::get<0>(ct_dims);
        ct_cols_loaded = std::get<1>(ct_dims);
        if (h_ct == nullptr || ct_cols_loaded <= 0) throw std::runtime_error("Failed to load coefficients.");
        ct_sparse_size = static_cast<size_t>(ct_rows_loaded) * ct_cols_loaded;
        size_t ct_sparse_bytes = ct_sparse_size * sizeof(float);
        std::cout << "Loaded " << ct_sparse_size << " sparse coefficients. Shape (" << ct_rows_loaded << "x" << ct_cols_loaded << ")" << std::endl;
        cudaMallocManaged(&d_ct, ct_sparse_bytes);
        memcpy(d_ct, h_ct, ct_sparse_bytes);
        delete[] h_ct; h_ct = nullptr;
        print_matrix(d_ct, 1, ct_sparse_size, 1, 20, "Sparse Coefficients (d_ct)");

        std::cout << "Loading locations from " << locs_directory << " into host memory..." << std::endl;
        auto locs_dims = load_ckpt_host_int(locs_directory, h_locs);
        locs_rows_loaded = std::get<0>(locs_dims);
        locs_cols_loaded = std::get<1>(locs_dims);
        if (h_locs == nullptr) throw std::runtime_error("Failed to load locations.");
        locs_sparse_size = static_cast<size_t>(locs_rows_loaded) * locs_cols_loaded;
        size_t locs_sparse_bytes = locs_sparse_size * sizeof(int);
        // Add sanity checks from original code here if desired
        std::cout << "Loaded " << locs_sparse_size << " location indices. Shape (" << locs_rows_loaded << "x" << locs_cols_loaded << ")" << std::endl;
        cudaMallocManaged(&d_locs, locs_sparse_bytes);
        memcpy(d_locs, h_locs, locs_sparse_bytes);
        delete[] h_locs; h_locs = nullptr;

        std::cout << "Allocating CUDA managed memory for dense float matrix d_ct_mat (" << ct_mat_bytes << " bytes)..." << std::endl;
        cudaMallocManaged(&d_ct_mat, ct_mat_bytes);
        std::cout << "Initializing dense matrix d_ct_mat to zero..." << std::endl;
        cudaMemset(d_ct_mat, 0, ct_mat_bytes);

        std::cout << "Reconstructing dense matrix d_ct_mat..." << std::endl;
        size_t num_elements_to_copy = std::min(ct_sparse_size, static_cast<size_t>(locs_cols_loaded));
        cudaDeviceSynchronize(); // Ensure managed memory accessible
        for (size_t i = 0; i < num_elements_to_copy; ++i) {
             int row_idx = -1;
             int col_idx = -1;
             if (locs_rows_loaded >= 2 && i < locs_cols_loaded && (static_cast<size_t>(locs_cols_loaded) + i) < locs_sparse_size) {
                 row_idx = d_locs[i];
                 col_idx = d_locs[locs_cols_loaded + i];
             } else { continue; }
             if (row_idx >= 0 && row_idx < ct_mat_rows && col_idx >= 0 && col_idx < ct_mat_cols) {
                 d_ct_mat[static_cast<size_t>(row_idx) * ct_mat_cols + col_idx] = d_ct[i];
             } else { /* Warning logic from original */ }
        }
        cudaDeviceSynchronize(); // Ensure writes visible
        std::cout << "Finished reconstructing dense matrix d_ct_mat." << std::endl;


        // --- Prepare for Inverse Fourier Transform (IFFT) ---

        // 10. Allocate PADDED Managed Memory for Complex Frequency Input
        std::cout << "\nAllocating PADDED CUDA managed memory for complex frequency input "
                  << "(d_freq_complex_padded, " << padded_complex_bytes << " bytes)..." << std::endl;
        cudaMallocManaged(&d_freq_complex_padded, padded_complex_bytes);
        cudaMemset(d_freq_complex_padded, 0, padded_complex_bytes); // IMPORTANT: Zero pad

        // 11. Copy Reconstructed Float Matrix to PADDED Complex Matrix (Imaginary=0) using Kernel
        std::cout << "Copying dense float matrix to PADDED complex frequency matrix "
                  << "(imaginary=0) using kernel..." << std::endl;
        dim3 threadsPerBlockPad(16, 16); // Example block size for 2D kernel
        dim3 numBlocksPad( (ct_mat_cols + threadsPerBlockPad.x - 1) / threadsPerBlockPad.x,
                           (ct_mat_rows + threadsPerBlockPad.y - 1) / threadsPerBlockPad.y );
        floatToComplexRealOnlyPadded<<<numBlocksPad, threadsPerBlockPad>>>(
            d_ct_mat, d_freq_complex_padded, ct_mat_rows, ct_mat_cols, padded_rows, padded_cols);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) throw std::runtime_error("CUDA Error after floatToComplexRealOnlyPadded: " + std::string(cudaGetErrorString(err)));
        cudaDeviceSynchronize();
        std::cout << "Finished copying to padded complex matrix." << std::endl;

        // --- Optional: Print the PADDED complex frequency matrix (for DEBUG) ---
        // print_complex_matrix(d_freq_complex_padded, padded_rows, padded_cols, 10, 10, "Padded Complex Freq Matrix");

        // 12. Allocate Managed Memory for Real Spatial Output (Original Size)
        size_t spatial_real_size = static_cast<size_t>(ct_mat_rows) * ct_mat_cols;
        size_t spatial_real_bytes = spatial_real_size * sizeof(float);
        std::cout << "Allocating CUDA managed memory for real spatial output (d_spatial_real)..." << std::endl;
        cudaMallocManaged(&d_spatial_real, spatial_real_bytes);

        // Allocate temporary buffer for transpose (Padded Size, transposed dimensions)
        size_t transpose_buffer_bytes = padded_mat_size * sizeof(Complex); // same size as padded input
        std::cout << "Allocating temporary buffer for transpose (d_temp_transpose)..." << std::endl;
        cudaMallocManaged(&d_temp_transpose, transpose_buffer_bytes);


        // --- Custom 2D IFFT Implementation ---
        std::cout << "\nExecuting Custom 2D IFFT..." << std::endl;

        // --- IFFT along Rows (on padded data) ---
        std::cout << " Step 1: IFFT along rows (" << padded_rows << " transforms of size " << padded_cols << ")..." << std::endl;
        int numBitsCols = static_cast<int>(log2(static_cast<float>(padded_cols)));
        if (!isPowerOfTwo(padded_cols)) throw std::runtime_error("Padded column dimension not power of 2!");

        // Launch one kernel per row. Each kernel uses shared memory.
        // WARNING: This launch configuration is simplified. A real implementation
        // might process multiple rows per block or use different strategies.
        // The ifft1D_kernel expects N threads for an N-sized transform.
        // We need shared memory size >= padded_cols * sizeof(Complex). Check limits.
        size_t sharedMemBytes = padded_cols * sizeof(Complex);
        int maxShared;
        cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlock, 0); // Get device capability
         if(sharedMemBytes > maxShared) {
            throw std::runtime_error("Required shared memory per block exceeds device limits for row FFT.");
         }

        for (int row = 0; row < padded_rows; ++row) {
            ifft1D_kernel<<<1, padded_cols, sharedMemBytes>>>(
                d_freq_complex_padded + static_cast<size_t>(row) * padded_cols, // Pointer to start of row
                padded_cols,
                numBitsCols
            );
             err = cudaGetLastError();
             if (err != cudaSuccess) throw std::runtime_error("CUDA Error during row IFFT kernel launch: " + std::string(cudaGetErrorString(err)));
        }
        cudaDeviceSynchronize(); // Wait for all row IFFTs
        std::cout << " Row IFFTs complete." << std::endl;

        // --- Transpose 1: Padded -> Temp ---
        std::cout << " Step 2: Transposing matrix (" << padded_rows << "x" << padded_cols << " -> " << padded_cols << "x" << padded_rows << ")..." << std::endl;
        dim3 threadsPerBlockTranspose(TILE_DIM, TILE_DIM);
        dim3 numBlocksTranspose( (padded_cols + TILE_DIM - 1) / TILE_DIM,
                                 (padded_rows + TILE_DIM - 1) / TILE_DIM );
        transposeComplex<<<numBlocksTranspose, threadsPerBlockTranspose>>>(
            d_freq_complex_padded, d_temp_transpose, padded_cols, padded_rows);
        err = cudaGetLastError();
        if (err != cudaSuccess) throw std::runtime_error("CUDA Error after transpose 1 kernel launch: " + std::string(cudaGetErrorString(err)));
        cudaDeviceSynchronize();
        std::cout << " Transpose 1 complete." << std::endl;

        // --- IFFT along Columns (now rows in d_temp_transpose) ---
        std::cout << " Step 3: IFFT along columns (now rows) (" << padded_cols << " transforms of size " << padded_rows << ")..." << std::endl;
        int numBitsRows = static_cast<int>(log2(static_cast<float>(padded_rows)));
         if (!isPowerOfTwo(padded_rows)) throw std::runtime_error("Padded row dimension not power of 2!");

        // Launch one kernel per "column" (which is now a row in d_temp_transpose)
        sharedMemBytes = padded_rows * sizeof(Complex);
         if(sharedMemBytes > maxShared) {
            throw std::runtime_error("Required shared memory per block exceeds device limits for column FFT.");
         }
        for (int col = 0; col < padded_cols; ++col) { // Iterate through original columns
             ifft1D_kernel<<<1, padded_rows, sharedMemBytes>>>(
                d_temp_transpose + static_cast<size_t>(col) * padded_rows, // Pointer to start of row in transposed matrix
                padded_rows,
                numBitsRows
            );
             err = cudaGetLastError();
             if (err != cudaSuccess) throw std::runtime_error("CUDA Error during column IFFT kernel launch: " + std::string(cudaGetErrorString(err)));
        }
        cudaDeviceSynchronize(); // Wait for all column IFFTs
        std::cout << " Column IFFTs complete." << std::endl;

        // --- Transpose 2: Temp -> Padded ---
        std::cout << " Step 4: Transposing matrix back (" << padded_cols << "x" << padded_rows << " -> " << padded_rows << "x" << padded_cols << ")..." << std::endl;
        // Need to swap width/height args for transpose kernel
        dim3 numBlocksTransposeBack( (padded_rows + TILE_DIM - 1) / TILE_DIM,
                                     (padded_cols + TILE_DIM - 1) / TILE_DIM );
        transposeComplex<<<numBlocksTransposeBack, threadsPerBlockTranspose>>>(
            d_temp_transpose, d_freq_complex_padded, padded_rows, padded_cols); // Width=padded_rows, Height=padded_cols (input dimensions for this call)
        err = cudaGetLastError();
        if (err != cudaSuccess) throw std::runtime_error("CUDA Error after transpose 2 kernel launch: " + std::string(cudaGetErrorString(err)));
        cudaDeviceSynchronize();
        std::cout << " Transpose 2 complete." << std::endl;

        // --- Extract Real Part, Crop, and Scale ---
        // The IFFT requires scaling by 1/(Total Elements). Use original size.
        float scale_factor = 1.0f / static_cast<float>(ct_mat_rows * ct_mat_cols); // Scale by ORIGINAL size
        std::cout << " Step 5: Extracting real part, cropping ("<< padded_rows << "x" << padded_cols << " -> " << ct_mat_rows << "x" << ct_mat_cols << ") and scaling (factor=" << scale_factor << ")..." << std::endl;

        dim3 threadsPerBlockExtract(16, 16); // Use 2D blocks matching the output structure
        dim3 numBlocksExtract( (ct_mat_cols + threadsPerBlockExtract.x - 1) / threadsPerBlockExtract.x,
                               (ct_mat_rows + threadsPerBlockExtract.y - 1) / threadsPerBlockExtract.y );

        extractRealAndScalePadded<<<numBlocksExtract, threadsPerBlockExtract>>>(
            d_freq_complex_padded, // Read from the final padded result
            d_spatial_real,        // Write to the original sized output
            ct_mat_rows,           // Original rows
            ct_mat_cols,           // Original cols
            padded_cols,           // Padded width for indexing input
            scale_factor
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) throw std::runtime_error("CUDA Error after extract/scale kernel launch: " + std::string(cudaGetErrorString(err)));
        cudaDeviceSynchronize();
        std::cout << "Custom IFFT and scaling finished." << std::endl;


        // --- Output the Result (delta(w)) ---
        std::cout << "\n--- Resulting Spatial Matrix (delta(w), managed memory) ---" << std::endl;
        print_matrix(d_spatial_real, ct_mat_rows, ct_mat_cols, 10, 10, "Delta(w) (Spatial Matrix)");
        std::cout << "----------------------------------------------------------\n" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        // Cleanup needs to include new buffers
        if (h_ct) delete[] h_ct;
        if (h_locs) delete[] h_locs;
        if (d_ct) cudaFree(d_ct);
        if (d_locs) cudaFree(d_locs);
        if (d_ct_mat) cudaFree(d_ct_mat);
        if (d_freq_complex_padded) cudaFree(d_freq_complex_padded);
        if (d_temp_transpose) cudaFree(d_temp_transpose);
        if (d_spatial_real) cudaFree(d_spatial_real);
        // Removed cufftDestroy
        return EXIT_FAILURE;
    }

    // --- Cleanup CUDA Managed Memory ---
    std::cout << "Cleaning up CUDA managed memory..." << std::endl;
    if (d_ct) cudaFree(d_ct);
    if (d_locs) cudaFree(d_locs);
    if (d_ct_mat) cudaFree(d_ct_mat);
    if (d_freq_complex_padded) cudaFree(d_freq_complex_padded);
    if (d_temp_transpose) cudaFree(d_temp_transpose);
    if (d_spatial_real) cudaFree(d_spatial_real);
    // Removed cufftDestroy

    std::cout << "Execution finished successfully. Delta(w) computed using custom IFFT." << std::endl;
    return 0;
}
