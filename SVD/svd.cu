#include <stdio.h>
#include <stdlib.h>

void get_args(int* m_p, int* n_p, int* r_p, int* num_threads_per_block_p, char* argv[]) {
    *m_p = strtol(argv[1], NULL, 10);
    *n_p = strtol(argv[2], NULL, 10);
    *r_p = strtol(argv[3], NULL, 10);
    *num_threads_per_block_p = strtol(argv[4], NULL, 10);

    if (*m_p <= 0 || *n_p <= 0 || *r_p <= 0 || *num_threads_per_block_p <= 0) {
        fprintf(stderr, "Error: m, n, r, and num_threads_per_block must be positive integers.\n");
        exit(EXIT_FAILURE);
    }
}

void memory_allocate(float** U, float** S, float** V, float** A, int m, int n, int r) {
    cudaMallocManaged((void**) U, m * r * sizeof(float));
    cudaMallocManaged((void**) S, r * sizeof(float));
    cudaMallocManaged((void**) V, n * r * sizeof(float));
    cudaMallocManaged((void**) A, m * n * sizeof(float));
    if (U == NULL || S == NULL || V == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        exit(EXIT_FAILURE);
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

void free_memory(float* U, float* S, float* V, float* A) {
    cudaFree(U);
    cudaFree(S);
    cudaFree(V);
    cudaFree(A);
}

__global__ void svd_kernel(float* U, float* S, float* V, float* A, int m, int n, int r) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float val = 0.0f;
        for (int _r = 0; _r < r; _r++) {
            float u = U[row * r + _r];
            float v = V[col * r + _r];
            val += S[_r] * u * v;
        }
        A[row * n + col] = val;
    }
}

void svd(float *U, float* S, float* V, float* A, int m, int n, int r, int num_threads_per_block) {
    dim3 threads_per_block(num_threads_per_block, num_threads_per_block);
    dim3 num_blocks((n + threads_per_block.x - 1) / threads_per_block.x, (m + threads_per_block.y - 1) / threads_per_block.y);
    printf("num_blocks = (%d, %d)\n", num_blocks.x, num_blocks.y);
    printf("threads_per_block = (%d, %d)\n", threads_per_block.x, threads_per_block.y);
    svd_kernel<<<num_blocks, threads_per_block>>>(U, S, V, A, m, n, r);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <m> <n> <r> <nt>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int m, n, r, num_threads_per_block;
    get_args(&m, &n, &r, &num_threads_per_block, argv);

    float *U, *S, *V, *A;
    memory_allocate(&U, &S, &V, &A, m, n, r);
    random_init(U, S, V, m, n, r);

    printf("U:\n");
    print_matrix(U, m, r);
    printf("S:\n");
    print_matrix(S, r, 1);
    printf("V:\n");
    print_matrix(V, n, r);

    svd(U, S, V, A, m, n, r, num_threads_per_block);

    printf("A (result of SVD):\n");
    print_matrix(A, m, n);

    free_memory(U, S, V, A);
}