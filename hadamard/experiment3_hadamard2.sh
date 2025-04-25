#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=llama2_7b_query_tokens_h2
#SBATCH --output="llama2_7b_query_tokens_h2.out"
#SBATCH --error="llama2_7b_query_tokens_h2.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --mem=32GB

# Load CUDA
module load nvidia/cuda/11.8.0

# Compile (optional)
nvcc hadamard.cu ./cnpy/cnpy.cpp ./utils/load_ckpt.cpp \
  -o hadamard_exec \
  -std=c++17 -O3 -Xcompiler "-Wall -Wno-unused-variable -Wno-unused-but-set-variable" \
  -I./utils -I./cnpy \
  -lz

# Fixed parameters
ROWS=4096
COLS=4096
TPB=1024
CT=./llama-2-7b/query/Llama-2-7b-hadamard-query-CT.npy
LOCS=./llama-2-7b/query/Llama-2-7b-hadamard-query-locs.npy

# Token variants
tokens=(1 16 128 512 1024)

# Run loop
for tok in $tokens; do
  INPUT="./llama-2-7b/inputs/x_${tok}.npy"
  echo "Running for tokens=$tok with TPB=$TPB"
  ./hadamard_exec $ROWS $COLS $tok $TPB $CT $LOCS $INPUT
done

