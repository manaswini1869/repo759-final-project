#! /usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=hadamard_cuda_query
#SBATCH --output="1_hadamard-gemma-2-2b-query.out"
#SBATCH --error="1_hadamard-gemma-2-2b-query.err"
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
  ./hadamard_exec 2304 2048 1 32 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_1.npy

  ./hadamard_exec 2304 2048 1 128 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_1.npy

  ./hadamard_exec 2304 2048 1 512 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_1.npy

  ./hadamard_exec 2304 2048 1 1024 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_1.npy

# Tokens = 16
  ./hadamard_exec 2304 2048 16 32 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_16.npy

  ./hadamard_exec 2304 2048 16 128 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_16.npy

  ./hadamard_exec 2304 2048 16 512 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_16.npy

  ./hadamard_exec 2304 2048 16 1024 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_16.npy

# Tokens = 128

  ./hadamard_exec 2304 2048 128 32 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_128.npy

  ./hadamard_exec 2304 2048 128 128 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_128.npy

  ./hadamard_exec 2304 2048 128 512 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_128.npy

  ./hadamard_exec 2304 2048 128 1024 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_128.npy

# Tokens = 512

  ./hadamard_exec 2304 2048 512 32 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_512.npy

  ./hadamard_exec 2304 2048 512 128 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_512.npy

  ./hadamard_exec 2304 2048 512 512 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_512.npy

  ./hadamard_exec 2304 2048 512 1024 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_512.npy

# Tokens = 1024

  ./hadamard_exec 2304 2048 1024 32 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_1024.npy

  ./hadamard_exec 2304 2048 1024 128 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_1024.npy

  ./hadamard_exec 2304 2048 1024 512 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_1024.npy

  ./hadamard_exec 2304 2048 1024 1024 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_1024.npy

   