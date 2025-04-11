#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J fourier-cu
#SBATCH -o %x.out
#SBATCH -e %x.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00
#SBATCH --mem=32GB

if [ "$1" = "compile" ]; then
  module load nvidia/cuda/11.8.0
  nvcc fourier.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o fourier-cu

elif [ "$1" = "run" ]; then
  for i in {10..30}; do
    rows=$((RANDOM % 100 + 1))
    cols=$((RANDOM % 100 + 1))
    n=$((2**i))
    ./fourier-cu $rows $cols $n
  done

elif [ "$1" = "clean" ]; then
  rm -f fourier-cu %x.out %x.err

else
  echo "./$0 [compile | run | clean]"
fi
