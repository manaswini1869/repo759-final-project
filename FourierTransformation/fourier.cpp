#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <fftw3.h>
#include <set>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <rows> <cols> <C>" << std::endl;
        return 1;
    }

    int rows = std::stoi(argv[1]);
    int cols = std::stoi(argv[2]);
    int C = std::stoi(argv[3]); // Ask what is to inserted for C?

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    // random frequency positions (shared across layers)
    std::set<std::pair<int, int>> freq_positions;
    while (freq_positions.size() < C) {
        int i = gen() % rows;
        int j = gen() % cols;
        freq_positions.insert({i, j});
    }

    // allocating complex frequency-domain matrix
    fftwf_complex* freq_data = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        freq_data[i][0] = 0.0f; // real part
        freq_data[i][1] = 0.0f; // imaginary part
    }

    // assign random values to selected spectral coefficients
    for (const auto& pos : freq_positions) {
        int i = pos.first;
        int j = pos.second;
        int idx = i * cols + j;
        freq_data[idx][0] = dist(gen); // real part
        freq_data[idx][1] = dist(gen); // imaginary part
    }

    // allocate output (spatial domain)
    float* spatial_data = (float*)fftwf_malloc(sizeof(float) * rows * cols);

    // creating and executing inverse FFT plan
    fftwf_plan plan = fftwf_plan_dft_c2r_2d(rows, cols, freq_data, spatial_data, FFTW_ESTIMATE);
    fftwf_execute(plan);

    // normalize and display result
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Reconstructed Spatial Domain Matrix (ΔW):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float value = spatial_data[i * cols + j] / (rows * cols); // normalize
            std::cout << std::setw(8) << value << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    fftwf_destroy_plan(plan);
    fftwf_free(freq_data);
    fftwf_free(spatial_data);


	// std::cout << "Original matrix (F):" << std::endl;
	// for (const auto& row : F) {
	// 	for (float val : row) {
	// 		std::cout << std::setw(6) << val << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// std::cout << "\nTransposed matrix (T):" << std::endl;
	// for (const auto& row : FT) {
	// 	for (float val : row) {
	// 		std::cout << std::setw(6) << val << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	return 0;
}
