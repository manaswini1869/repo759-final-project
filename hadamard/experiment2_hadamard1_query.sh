#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=hadamard_1024t
#SBATCH --output="hadamard_1024_tokens.out"
#SBATCH --error="hadamard_1024_tokens.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --mem=32GB

# Load CUDA
module load nvidia/cuda/11.8.0

# Compile (optional if already compiled)
nvcc hadamard.cu ./cnpy/cnpy.cpp ./utils/load_ckpt.cpp \
  -o hadamard_exec \
  -std=c++17 -O3 -Xcompiler "-Wall -Wno-unused-variable -Wno-unused-but-set-variable" \
  -I./utils -I./cnpy \
  -lz

# Model configs: (rows cols CT LOCS X_INPUT)
models=(
  "2304 2048 ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy ./gemma-2-2b/inputs/x_1024.npy"
  "3584 4096 ./gemma-2-9b/query/Gemma-2-9b-hadamard-query-CT.npy ./gemma-2-9b/query/Gemma-2-9b-hadamard-query-locs.npy ./gemma-2-9b/inputs/x_1024.npy"
  "4096 4096 ./llama-2-7b/query/Llama-2-7b-hadamard-query-CT.npy ./llama-2-7b/query/Llama-2-7b-hadamard-query-locs.npy ./llama-2-7b/inputs/x_1024.npy"
  "5120 5120 ./llama-2-13b/query/Llama-2-13b-hadamard-query-CT.npy ./llama-2-13b/query/Llama-2-13b-hadamard-query-locs.npy ./llama-2-13b/inputs/x_1024.npy"
  "4096 4096 ./llama-3-1-8b/query/Llama-3.1-8b-hadamard-query-CT.npy ./llama-3-1-8b/query/Llama-3.1-8b-hadamard-query-locs.npy ./llama-3-1-8b/inputs/x_1024.npy"
)

# Run loop
for model in "${models[@]}"; do
  set -- $model
  rows=$1
  cols=$2
  ct=$3
  locs=$4
  input=$5

  echo "Running model with rows=$rows cols=$cols"
  ./hadamard_exec $rows $cols 1024 1024 $ct $locs $input
done
