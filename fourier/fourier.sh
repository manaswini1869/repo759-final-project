#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J fourier
#SBATCH -o %x-test.out -e %x-test.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00
#SBATCH --mem=32GB


# Compilation command using nvcc
# -std=c++17: Use C++17 standard
# -o ${EXEC_NAME}: Specify the output executable name
# ${SRC_FILE}: The input CUDA source file
# ${LOAD_CKPT_OBJ}: Link against the pre-compiled object file for loading functions
# -lcufft: Link against the cuFFT library (needed due to cufft.h include)
# Add -I${CUDA_INCLUDE_PATH} and -L${CUDA_LIB_PATH} if CUDA paths are non-standard

module load nvidia/cuda/11.8.0
nvcc -std=c++17 fourier.cu ./cnpy/cnpy.cpp ./utils/load_ckpt.cpp -I./utils/ -I./cnpy -lcufft -lz -o fourier
./fourier ./Llama-2-7b/inputs/x_1024.npy 32 ./Gemma-2-2b/inputs/x_1024.npy ./Llama-2-7b/query/fourier/Llama-2-7b-fourier-query-CT.npy ./Llama-2-7b/query/fourier/Llama-2-7b-fourier-query-locs.npy
./fourier ./Llama-2-7b/inputs/x_1024.npy 128 ./Gemma-2-2b/inputs/x_1024.npy ./Llama-2-7b/query/fourier/Llama-2-7b-fourier-query-CT.npy ./Llama-2-7b/query/fourier/Llama-2-7b-fourier-query-locs.npy
./fourier ./Llama-2-7b/inputs/x_1024.npy 512 ./Gemma-2-2b/inputs/x_1024.npy ./Llama-2-7b/query/fourier/Llama-2-7b-fourier-query-CT.npy ./Llama-2-7b/query/fourier/Llama-2-7b-fourier-query-locs.npy
./fourier ./Llama-2-7b/inputs/x_1024.npy 1024 ./Gemma-2-2b/inputs/x_1024.npy ./Llama-2-7b/query/fourier/Llama-2-7b-fourier-query-CT.npy ./Llama-2-7b/query/fourier/Llama-2-7b-fourier-query-locs.npy

