#include <iostream>
#include "cnpy.h"
#include "load_ckpt.h"

std::tuple<size_t, size_t> load_ckpt_float(std::string& ckpt_directory, float*& data) {
    cnpy::NpyArray arr = cnpy::npy_load(ckpt_directory);

    size_t rows = arr.shape[0];
    size_t cols = arr.shape[1];
    if(cols == 0){
        cols = 1;
    }

    size_t total_elements = rows * cols;
    data = new float[total_elements];
    std::copy(arr.data<float>(), arr.data<float>() + total_elements, data);

    std::cout << "Loaded array shape: " << rows << "x" << cols << ", from " << ckpt_directory << "\n";

    return std::make_tuple(rows, cols);
}

std::tuple<size_t, size_t> load_ckpt_int(std::string& ckpt_directory, int*& data) {
    cnpy::NpyArray arr = cnpy::npy_load(ckpt_directory);

    size_t rows = arr.shape[0];
    size_t cols = arr.shape[1];
    if(cols == 0){
        cols = 1;
    }

    size_t total_elements = rows * cols;
    data = new int[total_elements];
    std::copy(arr.data<int>(), arr.data<int>() + total_elements, data);

    std::cout << "Loaded array shape: " << rows << "x" << cols << ", from " << ckpt_directory << "\n";

    return std::make_tuple(rows, cols);
}

void save_array(const std::string& filename, float*& data, size_t size) {
    cnpy::npy_save(filename, data, {size}, "w");
}

// template std::tuple<size_t, size_t> load_ckpt<float>(std::string& ckpt_directory, float*& data);
// template std::tuple<size_t, size_t> load_ckpt<int>(std::string& ckpt_directory, int*& data);