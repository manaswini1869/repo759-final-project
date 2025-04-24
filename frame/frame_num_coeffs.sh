#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J frame_layer_1_num_coeffs
#SBATCH -o results/%x-test.out -e results/%x-test.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00
#SBATCH --mem=32GB

module load nvidia/cuda/11.8.0

# nvcc -std=c++17 frame_layer_1.cu ../cnpy/cnpy.cpp ../utils/load_ckpt.cpp matmul.cu construct_frames.cpp -I../utils/ -I../cnpy -lz -o frame_layer_1

num_threads=1024
tokens=512
l=16

ncs=(1000 3000 5000 10000 16000)

for nc in ${ncs[@]}; do
	echo "Running with nc = $nc"

	coeffs_path=../checkpoints/num_coeffs/Llama-2-7b/query/frame/num_coeffs_${nc}/Llama-2-7b-frame-query-CT.npy
	locs_path=../checkpoints/num_coeffs/Llama-2-7b/query/frame/num_coeffs_${nc}/Llama-2-7b-frame-query-locs.npy
	inputs_path=../checkpoints/Llama-2-7b/inputs/x_512.npy

	./frame_layer_1 4096 4096 ${l} ${l} ${tokens} $num_threads $coeffs_path $locs_path $inputs_path
done

