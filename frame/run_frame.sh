#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J frame_layer_1_threads
#SBATCH -o %x-test.out -e %x-test.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00
#SBATCH --mem=32GB


module load nvidia/cuda/11.8.0

nvcc -std=c++17 frame_layer_1.cu ../cnpy/cnpy.cpp ../utils/load_ckpt.cpp matmul.cu construct_frames.cpp -I../utils/ -I../cnpy -lz -o frame_layer_1

num_threads=(32 128 512 1024)
coeffs_path=../checkpoints/Llama-2-7b/query/frame/Llama-2-7b-frame-query-CT.npy
locs_path=../checkpoints/Llama-2-7b/query/frame/Llama-2-7b-frame-query-locs.npy
inputs_path=../checkpoints/Llama-2-7b/inputs/x_1.npy
for threads in ${num_threads[@]}; do
    echo "Running with $threads threads"
    ./frame_layer_1 4096 4096 2 2 1 $threads $coeffs_path $locs_path $inputs_path
done

