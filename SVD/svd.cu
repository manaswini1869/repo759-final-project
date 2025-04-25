#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h> // Included for host timing if needed, but using CUDA events

// --- Kernel Implementation 1: W = sum(sigma_i * u_i * v_i^T) ---
__global__ void compute_sum_outer_products(const float* u,     // M x Nu
                                         const float* v,     // M x Nv
                                         const float* sigma, // M
                                         float* W_out,      // Nu x Nv (output)
                                         int M, int Nu, int Nv)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Column index for W (0..Nv-1)
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row index for W (0..Nu-1)

    if (j < Nu && k < Nv) {
        float sum_jk = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum_jk += sigma[i] * u[i * Nu + j] * v[i * Nv + k];
        }
        W_out[j * Nv + k] = sum_jk; // W is Nu x Nv
    }
}

// --- Kernel Implementation 1 & 2: Basic Matrix Multiply (Result = A * B) ---
// Used for Result1 = W * X (Impl 1)
__global__ void basic_matrix_multiply(const float* A, // In1: NumRowsA x NumColsA
                                    const float* B, // In2: NumColsA x NumColsB ( = NumRowsB)
                                    float* C,      // Out: NumRowsA x NumColsB
                                    int NumRowsA, int NumColsA, int NumColsB)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x; // Column index for C (0..NumColsB-1)
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row index for C (0..NumRowsA-1)

    if (j < NumRowsA && l < NumColsB) {
        float sum_jl = 0.0f;
        for (int k = 0; k < NumColsA; ++k) { // Inner dimension = NumColsA
            // A is NumRowsA x NumColsA, B is NumColsA x NumColsB
            sum_jl += A[j * NumColsA + k] * B[k * NumColsB + l];
        }
        C[j * NumColsB + l] = sum_jl; // C is NumRowsA x NumColsB
    }
}


// --- Kernel Implementation 2: r = v^T * X (computing all r_i rows) ---
// Each thread computes one element (r_i)_k
__global__ void compute_intermediate_rows(const float* v,     // M x Nv
                                          const float* X,     // Nv x P
                                          float* r_out,      // M x P (output)
                                          int M, int Nv, int P)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Column index for r (0..P-1)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index for r (0..M-1)

    if (i < M && k < P) {
        float sum_ik = 0.0f;
        for (int l = 0; l < Nv; ++l) { // Inner dimension = Nv
            // v is M x Nv, X is Nv x P
            sum_ik += v[i * Nv + l] * X[l * P + k];
        }
        r_out[i * P + k] = sum_ik; // r is M x P
    }
}

// --- Kernel Implementation 2: Result = sum(sigma_i * u_i * r_i) ---
// Each thread computes one element Result_jk
__global__ void compute_final_matrix_sum_from_rows(const float* u,       // M x Nu
                                                 const float* sigma,   // M
                                                 const float* r,       // M x P (from Kernel 2.1)
                                                 float* result_out, // Nu x P (output)
                                                 int M, int Nu, int P)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Column index for Result (0..P-1)
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row index for Result (0..Nu-1)

    if (j < Nu && k < P) {
        float sum_jk = 0.0f;
        for (int i = 0; i < M; ++i) {
            // u is M x Nu, r is M x P
            sum_jk += sigma[i] * u[i * Nu + j] * r[i * P + k];
        }
        result_out[j * P + k] = sum_jk; // Result is Nu x P
    }
}


// --- Host Code ---
int main(int argc, char **argv) {

    if (argc != 6) {
        fprintf(stderr, "Usage: %s <M> <Nu> <Nv> <P> <B>\n", argv[0]);
        fprintf(stderr, "  M:  Number of terms in sum\n");
        fprintf(stderr, "  Nu: Dimension of u vectors\n");
        fprintf(stderr, "  Nv: Dimension of v vectors (rows of X)\n");
        fprintf(stderr, "  P:  Columns of X\n");
        fprintf(stderr, "  B:  Block size\n");
        return 1;
    }

    // --- Get Dimensions ---
    int M = atoi(argv[1]);
    int Nu = atoi(argv[2]);
    int Nv = atoi(argv[3]);
    int P = atoi(argv[4]);
    int B = atoi(argv[5]);

    if (M <= 0 || Nu <= 0 || Nv <= 0 || P <= 0) {
        fprintf(stderr, "Error: All dimensions must be positive integers.\n");
        return 1;
    }

    // --- Data Sizes ---
    size_t u_size = (size_t)M * Nu * sizeof(float);
    size_t v_size = (size_t)M * Nv * sizeof(float);
    size_t sigma_size = (size_t)M * sizeof(float);
    size_t X_size = (size_t)Nv * P * sizeof(float);
    size_t W_size = (size_t)Nu * Nv * sizeof(float); // Impl 1 intermediate
    size_t r_size = (size_t)M * P * sizeof(float);  // Impl 2 intermediate
    size_t result_size = (size_t)Nu * P * sizeof(float); // Both results

    // --- Allocate Host Memory (Inputs Only) ---
    float *h_u = (float*)malloc(u_size);
    float *h_v = (float*)malloc(v_size);
    float *h_sigma = (float*)malloc(sigma_size);
    float *h_X = (float*)malloc(X_size);

    // --- Initialize Host Data ---
    for(size_t i = 0; i < (size_t)M * Nu; ++i) h_u[i] = (float)((i % 100) + 1) * 0.01f;
    for(size_t i = 0; i < (size_t)M * Nv; ++i) h_v[i] = (float)((i % 100) + 1) * 0.01f;
    for(size_t i = 0; i < M; ++i) h_sigma[i] = (float)(i + 1) * 0.1f;
    for(size_t i = 0; i < (size_t)Nv * P; ++i) h_X[i] = (float)((i % 100) + 1) * 0.01f;

    // --- Allocate Device Memory ---
    float *d_u, *d_v, *d_sigma, *d_X;
    float *d_W, *d_r; // Intermediates
    float *d_result1, *d_result2; // Outputs

    cudaMalloc((void**)&d_u, u_size);
    cudaMalloc((void**)&d_v, v_size);
    cudaMalloc((void**)&d_sigma, sigma_size);
    cudaMalloc((void**)&d_X, X_size);
    cudaMalloc((void**)&d_W, W_size);
    cudaMalloc((void**)&d_r, r_size);
    cudaMalloc((void**)&d_result1, result_size);
    cudaMalloc((void**)&d_result2, result_size);


    // --- Copy Inputs Host -> Device ---
    cudaMemcpy(d_u, h_u, u_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, v_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma, h_sigma, sigma_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, X_size, cudaMemcpyHostToDevice);

    // --- CUDA Events for Timing ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms;

    // --- Kernel Launch Configuration ---
    const int BLOCK_DIM_X = B;
    const int BLOCK_DIM_Y = B;
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);

    // --- Implementation 1: (sum_i sigma_i u_i v_i^T) * X ---
    cudaEventRecord(start);

    // Kernel 1.1: Compute W = sum(sigma_i * u_i * v_i^T)
    // Output W is Nu x Nv
    dim3 gridDimK1_1((Nv + blockDim.x - 1) / blockDim.x,
                     (Nu + blockDim.y - 1) / blockDim.y);
    compute_sum_outer_products<<<gridDimK1_1, blockDim>>>(d_u, d_v, d_sigma, d_W, M, Nu, Nv);

    // Kernel 1.2: Compute Result1 = W * X
    // Output Result1 is Nu x P. W is Nu x Nv, X is Nv x P
    dim3 gridDimK1_2((P + blockDim.x - 1) / blockDim.x,
                     (Nu + blockDim.y - 1) / blockDim.y);
    basic_matrix_multiply<<<gridDimK1_2, blockDim>>>(d_W, d_X, d_result1, Nu, Nv, P);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // Wait for kernels to complete
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("%.4f\n", time_ms);


    // --- Implementation 2: sum_i (sigma_i * (u_i * (v_i^T * X))) ---
    cudaEventRecord(start);

    // Kernel 2.1: Compute intermediate rows r_i = v_i^T * X
    // Output r is M x P
    dim3 gridDimK2_1((P + blockDim.x - 1) / blockDim.x,
                     (M + blockDim.y - 1) / blockDim.y);
    compute_intermediate_rows<<<gridDimK2_1, blockDim>>>(d_v, d_X, d_r, M, Nv, P);

    // Kernel 2.2: Compute Result2 = sum(sigma_i * u_i * r_i)
    // Output Result2 is Nu x P
    dim3 gridDimK2_2((P + blockDim.x - 1) / blockDim.x,
                     (Nu + blockDim.y - 1) / blockDim.y);
    compute_final_matrix_sum_from_rows<<<gridDimK2_2, blockDim>>>(d_u, d_sigma, d_r, d_result2, M, Nu, P);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // Wait for kernels to complete
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("%.4f\n", time_ms);


    // --- Cleanup ---
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_sigma);
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_r);
    cudaFree(d_result1);
    cudaFree(d_result2);

    free(h_u);
    free(h_v);
    free(h_sigma);
    free(h_X);

    return 0;
}