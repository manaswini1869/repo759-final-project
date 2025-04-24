#! /usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=lora
#SBATCH --output="lora.out"
#SBATCH --error="lora.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --mem=32GB

module load nvidia/cuda/11.8.0
nvcc lora.cu ./cnpy/cnpy.cpp ./utils/load_ckpt.cpp \
  -o lora \
  -std=c++17 -O3 -Xcompiler "-Wall -Wno-unused-variable -Wno-unused-but-set-variable" \
  -I./utils -I./cnpy \
  -lz
#nvcc -std c++17 lora.cu ./cnpy/cnpy.cpp ./utils/load_ckpt.cpp -I./utils/ -I./cnpy/ -Iz -Wall -Wno-unused-variable -Wno-unused-but-set-variable -O3 -Xcompiler -o lora
./lora
