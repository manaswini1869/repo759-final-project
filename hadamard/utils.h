// ==========================================
// utils.h
// ==========================================
#pragma once

#include <vector>
#include <string>

void load_sparse_C(const std::string& values_file, const std::string& locs_file,
                   std::vector<float>& values, std::vector<int>& locs, int& rows, int& cols);

void load_vector(const std::string& file, std::vector<float>& vec);

void save_vector(const std::string& file, const std::vector<float>& vec);
