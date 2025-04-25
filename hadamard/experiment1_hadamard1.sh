#! /usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=hadamard_cuda_query_1024
#SBATCH --output="7_hadamard-llama-2-13b-query-1024.out"
#SBATCH --error="7_hadamard-llama-2-13b-query-1024.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:15:00
#SBATCH --mem=32GB

# Load CUDA
module load nvidia/cuda/11.8.0

# Compile
nvcc hadamard.cu ./cnpy/cnpy.cpp ./utils/load_ckpt.cpp \
  -o hadamard_exec \
  -std=c++17 -O3 -Xcompiler "-Wall -Wno-unused-variable -Wno-unused-but-set-variable" \
  -I./utils -I./cnpy \
  -lz

# Run executions for tokens=1024 with various thread counts
for threads in 32 128 512 1024; do
  ./hadamard_exec 5120 5120 1024 $threads \
    ./llama-2-13b/query/Llama-2-13b-hadamard-query-CT.npy \
    ./llama-2-13b/query/Llama-2-13b-hadamard-query-locs.npy \
    ./llama-2-13b/inputs/x_1024.npy
done
