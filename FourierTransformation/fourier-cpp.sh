# !/usr/bin/env bash

# SBATCH -p instruction
# SBATCH -J fourier-cpp
# SBATCH -o %x.out -e %x.err
# SBATCH -c 20
# SBATCH -t 0-00:30:00

if [ "$1" = "compile" ]; then
  g++ fourier.cpp -Wall -O3 -std=c++11 -o fourier-cpp -fopenmp
	sudo apt install libfftw3-dev
elif [ "$1" = "run" ]; then
  for i in {10..12}; do
		rows=$((RANDOM % 100 + 1))
    cols=$((RANDOM % 100 + 1))
	  n=$((2**i))
	  ./fourier-cpp $rows $cols $n
  done
elif [ "$1" = "clean" ]; then
  rm -f fourier-cpp ${%x}.out ${%x}.err
else
  echo "./$0 [compile | run | clean]"
fi