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
./fourier ./inputs/x_1.npy 1
./fourier ./inputs/x_1.npy 2
./fourier ./inputs/x_1.npy 4
./fourier ./inputs/x_1.npy 8
./fourier ./inputs/x_1.npy 16
./fourier ./inputs/x_1.npy 72
./fourier ./inputs/x_1.npy 256
./fourier ./inputs/x_1.npy 1024


./fourier ./inputs/x_16.npy 1
./fourier ./inputs/x_16.npy 2
./fourier ./inputs/x_16.npy 4
./fourier ./inputs/x_16.npy 8
./fourier ./inputs/x_16.npy 16
./fourier ./inputs/x_16.npy 72
./fourier ./inputs/x_16.npy 256
./fourier ./inputs/x_16.npy 1024

./fourier ./inputs/x_128.npy 1
./fourier ./inputs/x_128.npy 2
./fourier ./inputs/x_128.npy 4
./fourier ./inputs/x_128.npy 8
./fourier ./inputs/x_128.npy 16
./fourier ./inputs/x_128.npy 72
./fourier ./inputs/x_128.npy 256
./fourier ./inputs/x_128.npy 1024

./fourier ./inputs/x_512.npy 1
./fourier ./inputs/x_512.npy 2
./fourier ./inputs/x_512.npy 4
./fourier ./inputs/x_512.npy 8
./fourier ./inputs/x_512.npy 16
./fourier ./inputs/x_512.npy 72
./fourier ./inputs/x_512.npy 256
./fourier ./inputs/x_512.npy 1024

./fourier ./inputs/x_1024.npy 1
./fourier ./inputs/x_1024.npy 2
./fourier ./inputs/x_1024.npy 4
./fourier ./inputs/x_1024.npy 8
./fourier ./inputs/x_1024.npy 16
./fourier ./inputs/x_1024.npy 72
./fourier ./inputs/x_1024.npy 256
./fourier ./inputs/x_1024.npy 1024
# Add -I/path/to/load_ckpt/header if load_ckpt.cuh is not in the same dir

