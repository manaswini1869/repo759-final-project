#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " <rows> <cols>" << std::endl;
		return 1;
	}

	int rows = std::stoi(argv[1]);
	int cols = std::stoi(argv[2]);
    int C = std::stoi(argv[3]);
	std::cout << C ;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

	std::vector<std::vector<float>> F(rows, std::vector<float>(cols));
    std::vector<std::vector<float>> FT(cols, std::vector<float>(rows)); // transpose matrix

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			F[i][j] = dist(gen);
		}
	}

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			FT[j][i] = F[i][j];
		}
	}

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
