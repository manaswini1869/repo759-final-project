#! /usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=lora
#SBATCH --output="lora.out"
#SBATCH --error="lora.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00

module load nvidia/cuda/11.8.0
nvcc -std=c++17 lora_update.cu ./cnpy/cnpy.cpp ./utils/load_ckpt.cpp -I./utils/ -I./cnpy/ -o lora_exec -lz
./lora_exec
