#include "utils.h"
#include <fstream>
#include <iostream>

void load_sparse_C(const std::string& values_file, const std::string& locs_file,
                   std::vector<float>& values, std::vector<int>& locs, int& rows, int& cols) {
    std::ifstream vfile(values_file, std::ios::binary);
    std::ifstream lfile(locs_file, std::ios::binary);

    if (!vfile.is_open() || !lfile.is_open()) {
        std::cerr << "Error opening files!\n";
        exit(1);
    }

    // Read values
    vfile.seekg(0, std::ios::end);
    size_t vsize = vfile.tellg() / sizeof(float);
    vfile.seekg(0, std::ios::beg);
    values.resize(vsize);
    vfile.read(reinterpret_cast<char*>(values.data()), vsize * sizeof(float));
    vfile.close();

    // Read locs (row, col pairs)
    lfile.seekg(0, std::ios::end);
    size_t lsize = lfile.tellg() / sizeof(int);
    lfile.seekg(0, std::ios::beg);
    locs.resize(lsize);
    lfile.read(reinterpret_cast<char*>(locs.data()), lsize * sizeof(int));
    lfile.close();

    // Infer rows and cols from locs
    rows = 0;
    cols = 0;
    for (size_t i = 0; i < locs.size(); i += 2) {
        if (locs[i] > rows) rows = locs[i];
        if (locs[i + 1] > cols) cols = locs[i + 1];
    }
    rows += 1;
    cols += 1;
}

void load_vector(const std::string& file, std::vector<float>& vec) {
    std::ifstream f(file, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Error opening file " << file << "!\n";
        exit(1);
    }
    f.seekg(0, std::ios::end);
    size_t size = f.tellg() / sizeof(float);
    f.seekg(0, std::ios::beg);
    vec.resize(size);
    f.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
    f.close();
}

void save_vector(const std::string& file, const std::vector<float>& vec) {
    std::ofstream f(file, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Error opening file " << file << "!\n";
        exit(1);
    }
    f.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(float));
    f.close();
}