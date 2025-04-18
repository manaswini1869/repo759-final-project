// your_program_name.cu
#include <algorithm> // For std::min
#include <cmath>     // For fabs in comparison
#include <cstring>   // For memcpy
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept> // For std::runtime_error
#include <string>    // For std::stoi, std::string
#include <vector>

// CUDA specific includes
#include <cuda_runtime.h> // For cudaMallocManaged, cudaMemset, cudaFree, cudaDeviceSynchronize, etc.
#include <cufft.h> // CUDA FFT library

#include "load_ckpt.cuh" // Assuming this header provides load_ckpt_host_float and load_ckpt_host_int

void print_matrix(const float *data, int rows, int cols, int print_rows = 10,
                  int print_cols = 10, const char *title = "Matrix") {
  std::cout << title << " (" << rows << "x" << cols << "):\n";
  int r_limit = std::min(rows, print_rows);
  int c_limit = std::min(cols, print_cols);
  if (!data) {
    std::cout << "(null pointer)\n";
    return;
  }
  // Ensure data is visible on the host for printing
  cudaDeviceSynchronize();

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

void print_complex_matrix(const cufftComplex *data, int rows, int cols,
                          int print_rows = 10, int print_cols = 10,
                          const char *title = "Complex Matrix") {
  std::cout << title << " (" << rows << "x" << cols << " complex elements):\n";
  int r_limit = std::min(rows, print_rows);
  int c_limit = std::min(cols, print_cols);
  if (!data) {
    std::cout << " (null pointer)\n";
    return;
  }
  // Ensure data is visible on the host for printing
  cudaDeviceSynchronize();

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

// --- CUDA Kernels ---

// Kernel to copy real data into the real part of complex data, setting
// imaginary to zero
__global__ void floatToComplexRealOnly(const float *real_in,
                                       cufftComplex *complex_out, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    complex_out[idx].x = real_in[idx];
    complex_out[idx].y = 0.0f;
  }
}

// Kernel to normalize real data after IFFT
__global__ void normalizeReal(float *data, size_t n, float scale_factor) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= scale_factor;
  }
}

int main(int argc, char *argv[]) {
  // Define the dimensions of the dense matrix
  int ct_mat_rows = 2304;
  int ct_mat_cols = 1024;
  size_t ct_mat_size = static_cast<size_t>(ct_mat_rows) * ct_mat_cols;
  size_t ct_mat_bytes =
      ct_mat_size * sizeof(float); // Size for the float matrix

  // Directories for loading data
  // NOTE: Adjust these paths based on where your executable is relative to the
  // checkpoint files.
  std::string ct_directory =
      "../checkpoints/gemma-2-2b/fourier/gemma-2-2b-fourier-CT.npy";
  std::string locs_directory =
      "../checkpoints/gemma-2-2b/fourier/gemma-2-2b-fourier-locs.npy";

  // Host pointers for loaded data
  float *h_ct = nullptr;
  int *h_locs = nullptr;

  // Managed memory pointers
  float *d_ct = nullptr;     // Managed memory for the sparse coefficients array
  int *d_locs = nullptr;     // Managed memory for the sparse locations array
  float *d_ct_mat = nullptr; // Managed memory for the reconstructed dense float
                             // matrix (Input to IFFT Prep)
  cufftComplex *d_freq_complex =
      nullptr; // Managed memory for the complex freq data (Input to IFFT)
  float *d_spatial_real = nullptr; // Managed memory for the spatial data
                                   // (Output of IFFT), i.e., delta(w)

  int ct_rows_loaded = 0, ct_cols_loaded = 0;
  int locs_rows_loaded = 0, locs_cols_loaded = 0;

  size_t ct_sparse_size = 0;   // Number of elements in the sparse ct array
  size_t locs_sparse_size = 0; // Number of elements in the sparse locs array

  cufftHandle plan; // cuFFT plan handle
  bool plan_created = false;

  try {
    // 1. Load Sparse Coefficients into Host Memory
    std::cout << "Loading coefficients from " << ct_directory
              << " into host memory..." << std::endl;
    auto ct_dims =
        load_ckpt_host_float(ct_directory, h_ct); // h_ct allocated by function
    ct_rows_loaded = std::get<0>(ct_dims);
    ct_cols_loaded = std::get<1>(ct_dims);
    if (h_ct == nullptr || ct_cols_loaded <= 0) {
      throw std::runtime_error(
          "Failed to load coefficients or zero coefficients loaded.");
    }
    ct_sparse_size = static_cast<size_t>(ct_rows_loaded) * ct_cols_loaded;
    size_t ct_sparse_bytes = ct_sparse_size * sizeof(float);
    std::cout << "Loaded " << ct_sparse_size << " sparse coefficients ("
              << ct_sparse_bytes << " bytes). Shape (" << ct_rows_loaded << "x"
              << ct_cols_loaded << ")" << std::endl;

    // 2. Allocate Managed Memory for Sparse Coefficients and Copy
    std::cout << "Allocating CUDA managed memory for sparse coefficients..."
              << std::endl;
    cudaMallocManaged(&d_ct, ct_sparse_bytes);
    std::cout << "Copying sparse coefficients from host to managed memory..."
              << std::endl;
    memcpy(d_ct, h_ct,
           ct_sparse_bytes); // Host-side copy is fine for managed memory

    // 3. Free Temporary Host Memory for Sparse Coefficients
    std::cout << "Freeing temporary host memory for sparse coefficients..."
              << std::endl;
    delete[] h_ct;
    h_ct = nullptr;

    std::cout << "\n--- First few loaded sparse coefficients (d_ct array, "
                 "managed memory) ---"
              << std::endl;
    print_matrix(d_ct, 1, ct_sparse_size, 1, 20, "Sparse Coefficients (d_ct)");
    std::cout << "-------------------------------------------------------------"
                 "-----------\n"
              << std::endl;

    // 4. Load Sparse Locations into Host Memory
    std::cout << "Loading locations from " << locs_directory
              << " into host memory..." << std::endl;
    auto locs_dims = load_ckpt_host_int(locs_directory,
                                        h_locs); // h_locs allocated by function
    locs_rows_loaded = std::get<0>(locs_dims);
    locs_cols_loaded = std::get<1>(locs_dims);
    if (h_locs == nullptr ||
        (locs_rows_loaded <= 0 &&
         locs_cols_loaded <=
             0)) { // Handle case where file exists but is empty/malformed
      throw std::runtime_error(
          "Failed to load locations or zero locations loaded.");
    }

    locs_sparse_size = static_cast<size_t>(locs_rows_loaded) * locs_cols_loaded;
    size_t locs_sparse_bytes = locs_sparse_size * sizeof(int);

    // Sanity checks on loaded dimensions vs expected
    if (ct_sparse_size !=
        locs_cols_loaded) { // Expect locs_cols_loaded to match the number of
                            // coefficients
      std::cerr << "Warning: Mismatch! Number of coefficients ("
                << ct_sparse_size << ") != number of location columns ("
                << locs_cols_loaded << "). Using minimum for reconstruction."
                << std::endl;
    }
    if (locs_rows_loaded != 2) {
      std::cerr << "Warning: Expected locations array to have 2 rows (row, col "
                   "indices), but got "
                << locs_rows_loaded
                << ". Assuming format [row_indices..., col_indices...]."
                << std::endl;
    }
    if (locs_sparse_size != 2 * locs_cols_loaded) {
      std::cerr << "Warning: Expected total locations elements to be 2 * "
                   "num_coefficients, but got "
                << locs_sparse_size << " vs " << 2 * locs_cols_loaded
                << ". Check locs file format." << std::endl;
    }

    std::cout << "Loaded " << locs_sparse_size << " location indices ("
              << locs_sparse_bytes << " bytes). Shape (" << locs_rows_loaded
              << "x" << locs_cols_loaded << ")" << std::endl;

    // 5. Allocate Managed Memory for Sparse Locations and Copy
    std::cout << "Allocating CUDA managed memory for sparse locations..."
              << std::endl;
    cudaMallocManaged(&d_locs, locs_sparse_bytes);
    std::cout << "Copying sparse locations from host to managed memory..."
              << std::endl;
    memcpy(d_locs, h_locs, locs_sparse_bytes);

    // 6. Free Temporary Host Memory for Locations
    std::cout << "Freeing temporary host memory for locations..." << std::endl;
    delete[] h_locs; // Assuming load_ckpt_host_int used new[]
    h_locs = nullptr;

    // 7. Allocate Managed Memory for the Dense Float Matrix (Reconstructed C)
    std::cout
        << "Allocating CUDA managed memory for dense float matrix d_ct_mat ("
        << ct_mat_bytes << " bytes)..." << std::endl;
    cudaMallocManaged(&d_ct_mat, ct_mat_bytes);

    // 8. Initialize Dense Matrix to Zero
    std::cout
        << "Initializing dense matrix d_ct_mat to zero using cudaMemset..."
        << std::endl;
    cudaMemset(d_ct_mat, 0, ct_mat_bytes);

    // 9. Reconstruct Dense Matrix (on Host, accessing Managed Memory)
    // Note: This reconstruction happens on the host using managed memory
    // pointers. For very large matrices, a kernel could be more efficient.
    std::cout << "Reconstructing dense matrix d_ct_mat on host using managed "
                 "memory pointers..."
              << std::endl;
    size_t num_elements_to_copy = std::min(
        ct_sparse_size,
        static_cast<size_t>(locs_cols_loaded)); // Use min of coefficients
                                                // loaded and location columns

    // Ensure loaded data is visible to the host
    cudaDeviceSynchronize();

    for (size_t i = 0; i < num_elements_to_copy; ++i) {
      int row_idx = -1;
      int col_idx = -1;

      // Assuming format where locs[i] is row, locs[locs_cols_loaded + i] is col
      if (locs_rows_loaded >= 2 && i < locs_cols_loaded &&
          (static_cast<size_t>(locs_cols_loaded) + i) < locs_sparse_size) {
        row_idx = d_locs[i];
        col_idx = d_locs[locs_cols_loaded +
                         i]; // Assumes [rows..., cols...] structure
      } else {
        // This case should ideally not be hit if locs file format is consistent
        // If it is hit, locs data is likely malformed or unexpected shape.
        if (i == 0)
          std::cerr << "Error: Unexpected location data format or insufficient "
                       "elements in locs array for coefficient "
                    << i << ". Expected at least " << 2 * locs_cols_loaded
                    << " elements for shape (2," << locs_cols_loaded << ")."
                    << std::endl;
        continue; // Skip this coefficient if location data is invalid
      }

      // Bounds check before writing to the dense matrix
      if (row_idx >= 0 && row_idx < ct_mat_rows && col_idx >= 0 &&
          col_idx < ct_mat_cols) {
        size_t dest_idx = static_cast<size_t>(row_idx) * ct_mat_cols + col_idx;
        d_ct_mat[dest_idx] = d_ct[i]; // Write to managed memory from host
      } else {
        if (i < 10) { // Print only a few warnings to avoid flooding console
          std::cerr << "Warning: Out-of-bounds location index (row=" << row_idx
                    << ", col=" << col_idx << ") for coefficient index " << i
                    << ". Skipping." << std::endl;
        } else if (i == 10) {
          std::cerr << "Warning: Further out-of-bounds warnings suppressed."
                    << std::endl;
        }
      }
    }

    // Ensure host writes to managed memory are visible to the device
    cudaDeviceSynchronize();
    std::cout << "Finished reconstructing dense matrix d_ct_mat." << std::endl;

    // --- Prepare for Inverse Fourier Transform (IFFT) ---

    // 10. Allocate Managed Memory for Complex Frequency Input
    // We assume the dense float matrix d_ct_mat represents the real parts of
    // the complex frequency matrix, with imaginary parts being zero.
    size_t freq_complex_size = static_cast<size_t>(ct_mat_rows) * ct_mat_cols;
    size_t freq_complex_bytes = freq_complex_size * sizeof(cufftComplex);
    std::cout << "\nAllocating CUDA managed memory for complex frequency input "
                 "(d_freq_complex)..."
              << std::endl;
    cudaMallocManaged(&d_freq_complex, freq_complex_bytes);

    // 11. Copy Reconstructed Float Matrix to Complex Matrix (Imaginary=0) using
    // Kernel
    std::cout << "Copying dense float matrix to complex frequency matrix "
                 "(imaginary=0) using kernel..."
              << std::endl;
    int threadsPerBlock = 256;
    int numBlocks = (freq_complex_size + threadsPerBlock - 1) / threadsPerBlock;
    floatToComplexRealOnly<<<numBlocks, threadsPerBlock>>>(
        d_ct_mat, d_freq_complex, freq_complex_size);
    cudaGetLastError();      // Check for kernel launch errors
    cudaDeviceSynchronize(); // Wait for kernel to finish
    std::cout << "Finished copying to complex matrix." << std::endl;

    // --- Optional: Print the complex frequency matrix (for DEBUG) ---
    if (ct_mat_rows <= 16 && ct_mat_cols <= 16) {
      std::cout << "\n--- Complex Frequency Matrix (d_freq_complex, managed "
                   "memory) ---"
                << std::endl;
      print_complex_matrix(d_freq_complex, ct_mat_rows, ct_mat_cols,
                           ct_mat_rows, ct_mat_cols,
                           "Complex Frequency Matrix");
      std::cout << "-----------------------------------------------------------"
                   "-------\n"
                << std::endl;
    } else {
      std::cout
          << "(Skipping print of large complex frequency matrix d_freq_complex)"
          << std::endl;
    }

    // 12. Allocate Managed Memory for Real Spatial Output (delta(w))
    size_t spatial_real_size =
        static_cast<size_t>(ct_mat_rows) *
        ct_mat_cols; // C2R IFFT of MxN complex gives MxN real
    size_t spatial_real_bytes = spatial_real_size * sizeof(float);
    std::cout << "Allocating CUDA managed memory for real spatial output "
                 "(d_spatial_real)..."
              << std::endl;
    cudaMallocManaged(&d_spatial_real, spatial_real_bytes);

    // 13. Create cuFFT Plan for 2D Complex-to-Real Inverse Transform
    std::cout << "Creating cuFFT 2D C2R inverse transform plan..." << std::endl;
    int dims[] = {ct_mat_cols,
                  ct_mat_rows}; // cuFFT dimensions are {nx, ny} = {cols, rows}
    cufftPlanMany(
        &plan,
        2,    // rank (2D)
        dims, // dimensions {nx, ny}
        NULL, 1,
        0, // Input layout (contiguous complex, istride=1, idist=0 for batch=1)
        NULL, 1,
        0, // Output layout (contiguous real, ostride=1, odist=0 for batch=1)
        CUFFT_C2R, // Transform type
        1);        // batch size
    plan_created = true;
    std::cout << "cuFFT plan created successfully." << std::endl;

    // 14. Execute the Inverse Fourier Transform
    std::cout << "Executing cuFFT C2R inverse transform..." << std::endl;
    cufftExecC2R(plan, d_freq_complex, d_spatial_real);
    cudaDeviceSynchronize(); // Wait for FFT to finish
    std::cout << "cuFFT execution finished." << std::endl;

    // 15. Apply Scaling Factor (1 / total_real_elements)
    // The C2R inverse transform output needs to be scaled by 1 / (rows * cols)
    float scale_factor = 1.0f / static_cast<float>(ct_mat_rows * ct_mat_cols);
    std::cout << "Applying scaling factor (" << scale_factor
              << ") to the output..." << std::endl;
    numBlocks = (spatial_real_size + threadsPerBlock - 1) /
                threadsPerBlock; // Recalculate if needed, but size is the same
                                 // as complex input size
    normalizeReal<<<numBlocks, threadsPerBlock>>>(
        d_spatial_real, spatial_real_size, scale_factor);
    cudaGetLastError();      // Check for kernel launch errors
    cudaDeviceSynchronize(); // Wait for kernel to finish
    std::cout << "Scaling finished." << std::endl;

    // --- Output the Result (delta(w)) ---

    std::cout << "\n--- Resulting Spatial Matrix (delta(w), managed memory) ---"
              << std::endl;
    // Print the top-left corner or the whole matrix if small
    print_matrix(d_spatial_real, ct_mat_rows, ct_mat_cols, 10, 10,
                 "Delta(w) (Spatial Matrix)");
    std::cout << "----------------------------------------------------------\n"
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "An error occurred: " << e.what() << std::endl;
    // Cleanup any allocated memory before exiting
    if (h_ct)
      delete[] h_ct;
    if (h_locs)
      delete[] h_locs;
    if (d_ct)
      cudaFree(d_ct);
    if (d_locs)
      cudaFree(d_locs);
    if (d_ct_mat)
      cudaFree(d_ct_mat);
    if (d_freq_complex)
      cudaFree(d_freq_complex);
    if (d_spatial_real)
      cudaFree(d_spatial_real);
    if (plan_created)
      cufftDestroy(plan); // Destroy plan if created
    return EXIT_FAILURE;
  }

  // --- Cleanup CUDA Managed Memory and cuFFT Plan ---
  std::cout << "Cleaning up CUDA managed memory and cuFFT plan..." << std::endl;
  if (d_ct)
    cudaFree(d_ct);
  if (d_locs)
    cudaFree(d_locs);
  if (d_ct_mat)
    cudaFree(d_ct_mat);
  if (d_freq_complex)
    cudaFree(d_freq_complex);
  if (d_spatial_real)
    cudaFree(d_spatial_real);
  if (plan_created)
    cufftDestroy(plan);

  std::cout << "Execution finished successfully. Delta(w) computed."
            << std::endl;
  return 0;
}
