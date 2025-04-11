#include "vscale.cuh"
#include <cuda.h>
#include <stdio.h>
#include <random>

#define NUM_THREADS 512

int main(int argc, char *argv[]) {
	if (argc < 3) {
		printf("Usage: %s <rows> <cols>\n", argv[0]);
		return 1;
	}

	int rows = atoi(argv[1]);
	int cols = atoi(argv[2]);
    int C = atoi(argv[3]);
	int total_size = rows * cols;

	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	const float minF = -10.0f, maxF = 10.0f;
	std::uniform_real_distribution<float> distF(minF, maxF);

	float *F;
	cudaMallocManaged((void **)&F, sizeof(float) * total_size);

    float *FT; // transpose matrix
	cudaMallocManaged((void **)&T, sizeof(float) * total_size);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			F[i * cols + j] = distF(generator);
		}
	}

    for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			FT[j * rows + i] = F[i * cols + j];
		}
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int device = -1;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(F, sizeof(float) * total_size, device, NULL);

	int NUM_BLOCKS = (total_size + NUM_THREADS - 1) / NUM_THREADS;

	cudaEventRecord(start);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	// Cleanup
	cudaFree(F);
}
