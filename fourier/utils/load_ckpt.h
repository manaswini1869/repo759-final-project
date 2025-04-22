#ifndef LOAD_CKPT_H
#define LOAD_CKPT_H

#include <string>
#include <tuple>

// Function declaration for load_ckpt
std::tuple<size_t, size_t> load_ckpt_float(std::string& ckpt_directory, float*& data);
std::tuple<size_t, size_t> load_ckpt_int(std::string& ckpt_directory, int*& data);

// Function declaration for save_array
void save_array(const std::string& filename, float*& data, size_t size);

#endif // LOAD_CKPT_H