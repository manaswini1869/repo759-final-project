#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J frame_layer_3_block_size
#SBATCH -o results/%x-test.out -e results/%x-test.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00
#SBATCH --mem=32GB

module load nvidia/cuda/11.8.0

# nvcc -std=c++17 frame_layer_1.cu ../cnpy/cnpy.cpp ../utils/load_ckpt.cpp matmul.cu construct_frames.cpp -I../utils/ -I../cnpy -lz -o frame_layer_1

num_threads=1024
tokens=512

ls=(2 4 16 64 256)


for l in ${ls[@]}; do
	echo "Running with block_size $l"

	coeffs_path=../checkpoints/block_size_ct/Llama-2-7b/query/frame/block_size_${l}/Llama-2-7b-frame-query-CT.npy
	locs_path=../checkpoints/block_size_ct/Llama-2-7b/query/frame/block_size_${l}/Llama-2-7b-frame-query-locs.npy
	inputs_path=../checkpoints/Llama-2-7b/inputs/x_512.npy

	./frame_layer_3 4096 4096 ${l} ${l} ${tokens} $num_threads $coeffs_path $locs_path $inputs_path
done

