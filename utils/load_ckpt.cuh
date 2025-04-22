#ifndef LOAD_CKPT_CUH
#define LOAD_CKPT_CUH

#include <string>
#include <tuple>
#include <cstddef> // For size_t

// ----- Function Declarations -----

/**
 * @brief Loads float data from a .npy file using cnpy into host memory.
 *
 * Allocates memory using 'new float[]'. The caller is responsible for 'delete[]'.
 *
 * @param ckpt_path Path to the .npy file.
 * @param data Reference to a float pointer. On success, this will point to the newly allocated host memory containing the data.
 * @return A tuple containing the number of rows and columns (size_t, size_t) of the loaded array.
 * Returns (rows, 1) for 1D arrays.
 */
std::tuple<size_t, size_t> load_ckpt_host_float(const std::string& ckpt_path, float*& data);

/**
 * @brief Loads int data from a .npy file using cnpy into host memory.
 *
 * Allocates memory using 'new int[]'. The caller is responsible for 'delete[]'.
 *
 * @param ckpt_path Path to the .npy file.
 * @param data Reference to an int pointer. On success, this will point to the newly allocated host memory containing the data.
 * @return A tuple containing the number of rows and columns (size_t, size_t) of the loaded array.
 * Returns (rows, 1) for 1D arrays.
 */
std::tuple<size_t, size_t> load_ckpt_host_int(const std::string& ckpt_path, int*& data);

/**
 * @brief Saves float data from host memory to a .npy file using cnpy.
 *
 * @param filename The desired output path for the .npy file.
 * @param data Pointer to the host float data to save.
 * @param rows Number of rows (first dimension).
 * @param cols Number of columns (second dimension, use 1 for 1D arrays).
 */
void save_array_host_float(const std::string& filename, const float* data, size_t rows, size_t cols = 1);


#endif // LOAD_CKPT_CUH