#! /usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=hadamard_cuda
#SBATCH --output="hadamard_%j.out"
#SBATCH --error="hadamard_%j.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:05:00

# Load CUDA module
module load nvidia/cuda/11.8.0

# Compile all cpp/cu files into a single executable
nvcc main.cu matrix_ops.cu utils.cpp -o hadamard_exec -std=c++17 -O3 -Xcompiler -Wall

# Loop over all inputs starting with x_*.bin
for xfile in inputs/x_*.bin
do
    echo "Running for input file: $xfile"
    ./hadamard_exec "$xfile"
done

