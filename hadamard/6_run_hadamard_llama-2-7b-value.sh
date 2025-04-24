#! /usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=hadamard_cuda_value
#SBATCH --output="6_hadamard-llama-2-7b-value.out"
#SBATCH --error="6_hadamard-llama-2-7b-value.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --mem=32GB

# Load CUDA
module load nvidia/cuda/11.8.0

# Compile
nvcc hadamard.cu ./cnpy/cnpy.cpp ./utils/load_ckpt.cpp \
  -o hadamard_exec \
  -std=c++17 -O3 -Xcompiler "-Wall -Wno-unused-variable -Wno-unused-but-set-variable" \
  -I./utils -I./cnpy \
  -lz


# Execute with correct argument order
# Format: <ct_rows> <ct_cols> <num_tokens> <threads_per_block> <CT> <LOCS> <X>
# Tokens = 1
  ./hadamard_exec 4096 4096 1 32 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_1.npy

  ./hadamard_exec 4096 4096 1 128 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_1.npy

  ./hadamard_exec 4096 4096 1 512 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_1.npy

  ./hadamard_exec 4096 4096 1 1024 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_1.npy

# Tokens = 16
  ./hadamard_exec 4096 4096 16 32 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_16.npy

  ./hadamard_exec 4096 4096 16 128 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_16.npy

  ./hadamard_exec 4096 4096 16 512 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_16.npy

  ./hadamard_exec 4096 4096 16 1024 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_16.npy

# Tokens = 128

  ./hadamard_exec 4096 4096 128 32 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_128.npy

  ./hadamard_exec 4096 4096 128 128 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_128.npy

  ./hadamard_exec 4096 4096 128 512 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_128.npy

  ./hadamard_exec 4096 4096 128 1024 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_128.npy

# Tokens = 512

  ./hadamard_exec 4096 4096 512 32 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_512.npy

  ./hadamard_exec 4096 4096 512 128 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_512.npy

  ./hadamard_exec 4096 4096 512 512 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_512.npy

  ./hadamard_exec 4096 4096 512 1024 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_512.npy

# Tokens = 1024

  ./hadamard_exec 4096 4096 1024 32 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_1024.npy

  ./hadamard_exec 4096 4096 1024 128 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_1024.npy

  ./hadamard_exec 4096 4096 1024 512 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_1024.npy

  ./hadamard_exec 4096 4096 1024 1024 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy \
  ./llama-2-7b/inputs/x_1024.npy

   