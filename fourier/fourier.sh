#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J fourier
#SBATCH -o %x-512.out -e %x-512.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00
#SBATCH --mem=32GB

if [ "$1" = "compile" ]; then
  module load nvidia/cuda/11.8.0
#   nvcc fourier.cu -lcufft -o fourier
    nvcc fourier.cu -o fourier -Xcompiler -O3 -Xcompiler -Wall -Xptxas -lcufft -std=c++17
elif [ "$1" = "run" ]; then
    for i in {10..29}; do
    n=$((2**i))
    ./fourier $n 128 512
done
elif [ "$1" = "clean" ]; then
  rm -f task2-512 task2-512.out task2-512.err
else
  echo "./$0 [compile | run | clean]"
fi