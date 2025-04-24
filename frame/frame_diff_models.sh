#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J frame_layer_3_layers_value
#SBATCH -o results/%x-test.out -e results/%x-test.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:30:00
#SBATCH --mem=32GB


module load nvidia/cuda/11.8.0

# nvcc -std=c++17 frame_layer_3.cu ../cnpy/cnpy.cpp ../utils/load_ckpt.cpp matmul.cu construct_frames.cpp -I../utils/ -I../cnpy -lz -o frame_layer_3

num_threads=1024
num_tokens=512
tff_l=2
layer=value
models=(Llama-2-7b Llama-2-13b Gemma-2-2b Gemma-2-9b Llama-3.1-8b)
inp_dims=(4096 5120 2304 3584 4096)
out_dims=(4096 5120 2048 4096 4096)

for idx in ${!models[@]}; do
	model=${models[$idx]}
	inp_dim=${inp_dims[$idx]}
	out_dim=${out_dims[$idx]}
	echo "Running with model=${model} inp_dim=${inp_dim} out_dim=${out_dim} l=${tff_l} n=${num_tokens} num_threads=${num_threads}"
	coeffs_path=../checkpoints/${model}/${layer}/frame/${model}-frame-${layer}-CT.npy
	locs_path=../checkpoints/${model}/${layer}/frame/${model}-frame-${layer}-locs.npy
	inputs_path=../checkpoints/${model}/inputs/x_${num_tokens}.npy
	./frame_layer_3 ${inp_dim} ${out_dim} ${tff_l} ${tff_l} ${num_tokens} ${num_threads} ${coeffs_path} ${locs_path} ${inputs_path}
done

