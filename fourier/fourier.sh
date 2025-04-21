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
nvcc -std=c++17 fourier-test.cu ../cnpy/cnpy.cpp ../utils/load_ckpt.cu -I../utils -I../cnpy -lcufft -lz -o fourier
./fourier ./inputs/x_16.npy
# Add -I/path/to/load_ckpt/header if load_ckpt.cuh is not in the same dir

