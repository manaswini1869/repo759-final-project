#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=hadamard2_value_1024
#SBATCH --output="hadamard2_value_1024.out"
#SBATCH --error="hadamard2_value_1024.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --mem=32GB

# Load CUDA
module load nvidia/cuda/11.8.0

# Compile (optional)
nvcc hadamard2.cu ./cnpy/cnpy.cpp ./utils/load_ckpt.cpp \
  -o hadamard2_exec \
  -std=c++17 -O3 -Xcompiler "-Wall -Wno-unused-variable -Wno-unused-but-set-variable" \
  -I./utils -I./cnpy \
  -lz

# Model configs: (rows cols CT LOCS X_INPUT) — VALUE layers
models=(
  "2304 1024 ./gemma-2-2b/value/Gemma-2-2b-hadamard-value-CT.npy ./gemma-2-2b/value/Gemma-2-2b-hadamard-value-locs.npy ./gemma-2-2b/inputs/x_1024.npy"
  "3584 2048 ./gemma-2-9b/value/Gemma-2-9b-hadamard-value-CT.npy ./gemma-2-9b/value/Gemma-2-9b-hadamard-value-locs.npy ./gemma-2-9b/inputs/x_1024.npy"
  "4096 4096 ./llama-2-7b/value/Llama-2-7b-hadamard-value-CT.npy ./llama-2-7b/value/Llama-2-7b-hadamard-value-locs.npy ./llama-2-7b/inputs/x_1024.npy"
  "5120 5120 ./llama-2-13b/value/Llama-2-13b-hadamard-value-CT.npy ./llama-2-13b/value/Llama-2-13b-hadamard-value-locs.npy ./llama-2-13b/inputs/x_1024.npy"
  "4096 4096 ./llama-3-1-8b/value/Llama-3.1-8b-hadamard-value-CT.npy ./llama-3-1-8b/value/Llama-3.1-8b-hadamard-value-locs.npy ./llama-3-1-8b/inputs/x_1024.npy"
)

# Run loop
for model in "${models[@]}"; do
  set -- $model
  rows=$1
  cols=$2
  ct=$3
  locs=$4
  input=$5

  echo "Running VALUE layer: rows=$rows cols=$cols"
  ./hadamard2_exec $rows $cols 1024 1024 $ct $locs $input
done
