#include <stdio.h>
#include <stdlib.h>

void get_args(m_p, n_p, r_p, num_threads_per_block, argv) {
    *m_p = strtol(argv[1], NULL, 10);
    *n_p = strtol(argv[2], NULL, 10);
    *r_p = strtol(argv[3], NULL, 10);
    *num_threads_per_block = strtol(argv[4], NULL, 10);

    if (m <= 0 || n <= 0 || r <= 0 || num_threads_per_block <= 0) {
        fprintf(stderr, "Error: m, n, r, and num_threads_per_block must be positive integers.\n");
        return EXIT_FAILURE;
    }
}

void memory_allocate(float** U, float** S, float** V, float** A, int m, int n, int r) {
    cudaMallocManaged((void**) &U, m * r * sizeof(float));
    cudaMallocManaged((void**) &S, r * sizeof(float));
    cudaMallocManaged((void**) &V, n * r * sizeof(float));
    cudaMallocManaged((void**) &A, m * n * sizeof(float));
    if (U == NULL || S == NULL || V == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return EXIT_FAILURE;
    }
}

void random_init(float* U, float* S, float* V, int m, int n, int r) {
    for (int i = 0; i < m * r; i++)
        U[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < r; i++)
        S[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < n * r; i++)
        V[i] = (float)rand() / RAND_MAX;
}

void print_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void print_vector(float* vector, int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ", vector[i]);
    }
    printf("\n");
}

__global__ void svd_kernel(float* U, float* S, float* V, float* A, int m, int n, int r) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float val = 0.0f;
        for (int r = 0; r < R; ++r) {
            float u = U[row * R + r];
            float v = V[col * R + r];
            val += S[r] * u * v;
        }
        A[row * n + col] = val;
    }
}

void svd(float *U, float* S, float* V, float* A, int m, int n, int r, int num_threads_per_block) {
    dim3 threads_per_block(num_threads_per_block, num_threads_per_block);
    dim3 num_blocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    svd_kernel<<<num_bocks, threads_per_block>>>(U, S, V, A, m, n, r);
}

void free_memory(float* U, float* S, float* V, float* A) {
    cudaFree(U);
    cudaFree(S);
    cudaFree(V);
    cudaFree(A);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <m> <n> <r> <nt>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int m, n, r, num_threads_per_block;
    get_args(&m, &n, &r, &num_threads_per_block, argv);

    float *U, *S, *V, *A;
    memory_allocate(&U, &S, &V, &A, m, n, r);
    random_init(U, S, V, m, n, r);

    svd(U, S, V, A, m, n, r, num_threads_per_block);
    cudaDeviceSynchronize();

    free_memory(U, S, V, A);
}