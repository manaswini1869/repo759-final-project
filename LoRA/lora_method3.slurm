#! /usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=method3wx
#SBATCH --output="method3wx.out"
#SBATCH --error="method3wx.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00

module load nvidia/cuda/11.8.0
nvcc method3wx.cu ./cnpy/cnpy.cpp ./utils/load_ckpt.cpp \
  -o method3wx \
  -std=c++17 -O3 -Xcompiler "-Wall -Wno-unused-variable -Wno-unused-but-set-variable" \
  -I./utils -I./cnpy \
  -lz
#nvcc method3wx.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o method3wx
./method3wx
