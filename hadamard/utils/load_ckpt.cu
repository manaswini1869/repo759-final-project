#include "load_ckpt.cuh" // Include the corresponding header
#include "cnpy.h"        // Include the dependency
#include <iostream>
#include <vector>        // cnpy might use std::vector internally or for shape
#include <numeric>       // For std::accumulate if needed, or other algorithms
#include <stdexcept>     // For throwing errors
#include <algorithm>     // For std::copy

// ----- Function Definitions -----

std::tuple<size_t, size_t> load_ckpt_host_float(const std::string& ckpt_path, float*& data) {
    try {
        cnpy::NpyArray arr = cnpy::npy_load(ckpt_path);

        // Basic validation
        if (arr.word_size != sizeof(float)) {
             throw std::runtime_error("Data type mismatch in NPY file: expected float (size "
                + std::to_string(sizeof(float)) + "), got size " + std::to_string(arr.word_size));
        }
        if (arr.shape.empty()) {
            throw std::runtime_error("NPY file has empty shape: " + ckpt_path);
        }

        size_t rows = arr.shape[0];
        // Handle 1D vs 2D+ shapes robustly
        size_t cols = (arr.shape.size() > 1) ? arr.shape[1] : 1;
        size_t higher_dims_product = 1;
        if (arr.shape.size() > 2) {
             // If more than 2D, flatten remaining dimensions into 'cols' conceptually
             // Or throw an error if only 1D/2D is expected.
             // Here, we flatten: cols *= arr.shape[2] * arr.shape[3] * ...
             for(size_t i = 2; i < arr.shape.size(); ++i) {
                 higher_dims_product *= arr.shape[i];
             }
             cols *= higher_dims_product;
             // Alternatively, error out:
             // throw std::runtime_error("Only 1D or 2D arrays supported for loading.");
        }


        size_t total_elements = rows * cols;
        if (arr.num_vals != total_elements) {
             throw std::runtime_error("NPY file shape inconsistent with num_vals: " + ckpt_path);
        }

        // Allocate HOST memory
        try {
             data = new float[total_elements];
        } catch (const std::bad_alloc& e) {
             std::cerr << "Failed to allocate host memory (" << total_elements * sizeof(float)
                       << " bytes) for NPY data: " << e.what() << std::endl;
             throw; // Re-throw the exception
        }


        // Copy data from cnpy buffer to the newly allocated host buffer
        const float* src_data = arr.data<float>();
        if (!src_data) {
            delete[] data; // Clean up allocated memory before throwing
            data = nullptr;
            throw std::runtime_error("Failed to get float data pointer from NPY array: " + ckpt_path);
        }
        std::copy(src_data, src_data + total_elements, data);

        std::cout << "Loaded host array shape: " << rows << "x" << cols << " (" << total_elements << " floats), from " << ckpt_path << "\n";

        // arr object goes out of scope here, releasing its internally held data buffer.

        return std::make_tuple(rows, cols);

    } catch (const std::exception& e) {
        std::cerr << "Error loading NPY file '" << ckpt_path << "': " << e.what() << std::endl;
        data = nullptr; // Ensure data is null on error
        // Re-throw or handle appropriately, maybe return {0, 0}
        throw; // Re-throwing for caller to handle
        // return std::make_tuple(0, 0); // Alternative: return zero size
    }
}

std::tuple<size_t, size_t> load_ckpt_host_int(const std::string& ckpt_path, int*& data) {
     try {
        cnpy::NpyArray arr = cnpy::npy_load(ckpt_path);

        // Basic validation
        if (arr.word_size != sizeof(int)) {
             throw std::runtime_error("Data type mismatch in NPY file: expected int (size "
                + std::to_string(sizeof(int)) + "), got size " + std::to_string(arr.word_size));
        }
         if (arr.shape.empty()) {
            throw std::runtime_error("NPY file has empty shape: " + ckpt_path);
        }

        size_t rows = arr.shape[0];
        size_t cols = (arr.shape.size() > 1) ? arr.shape[1] : 1;
        size_t higher_dims_product = 1;
        if (arr.shape.size() > 2) {
             for(size_t i = 2; i < arr.shape.size(); ++i) {
                 higher_dims_product *= arr.shape[i];
             }
             cols *= higher_dims_product;
        }

        size_t total_elements = rows * cols;
         if (arr.num_vals != total_elements) {
             throw std::runtime_error("NPY file shape inconsistent with num_vals: " + ckpt_path);
        }

        // Allocate HOST memory
        try {
             data = new int[total_elements];
        } catch (const std::bad_alloc& e) {
             std::cerr << "Failed to allocate host memory (" << total_elements * sizeof(int)
                       << " bytes) for NPY data: " << e.what() << std::endl;
             throw;
        }


        // Copy data from cnpy buffer to the newly allocated host buffer
        const int* src_data = arr.data<int>();
         if (!src_data) {
            delete[] data;
            data = nullptr;
            throw std::runtime_error("Failed to get int data pointer from NPY array: " + ckpt_path);
        }
        std::copy(src_data, src_data + total_elements, data);

        std::cout << "Loaded host array shape: " << rows << "x" << cols << " (" << total_elements << " ints), from " << ckpt_path << "\n";

        return std::make_tuple(rows, cols);

    } catch (const std::exception& e) {
        std::cerr << "Error loading NPY file '" << ckpt_path << "': " << e.what() << std::endl;
        data = nullptr;
        throw;
        // return std::make_tuple(0, 0);
    }
}

void save_array_host_float(const std::string& filename, const float* data, size_t rows, size_t cols) {
    if (!data) {
        std::cerr << "Error saving array: data pointer is null." << std::endl;
        return; // Or throw
    }
    if (rows == 0 || cols == 0) {
         std::cerr << "Error saving array: dimensions cannot be zero (rows=" << rows << ", cols=" << cols << ")." << std::endl;
        return; // Or throw
    }

    std::vector<size_t> shape;
    if (cols == 1 && rows > 1) { // Treat as 1D if cols is 1 (and rows > 1)
        shape = {rows};
    } else if (rows == 1 && cols > 1) { // Treat as 1D if rows is 1 (and cols > 1)
         shape = {cols};
    } else { // Treat as 2D
        shape = {rows, cols};
    }

    try {
        // Note: cnpy::npy_save expects a non-const pointer for its template deduction,
        // but it doesn't modify the data when mode is "w". We use const_cast carefully.
        cnpy::npy_save(filename, const_cast<float*>(data), shape, "w");
        std::cout << "Saved host array to " << filename << " with shape ";
        for(size_t i=0; i<shape.size(); ++i) std::cout << shape[i] << (i == shape.size()-1 ? "" : "x");
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error saving NPY file '" << filename << "': " << e.what() << std::endl;
        // Handle error appropriately
    }
}