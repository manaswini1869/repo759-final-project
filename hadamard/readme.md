This part of the course project implements GPU-accelerated matrix operations using Hadamard transformations in CUDA. 

The computation flow includes:
Step1: Constructing a sparse matrix C from .npy files (values, locations)
Step2: Generating Hadamard matrices H and Hᵗ
Step3: Calculating ΔW = H × C × Hᵗ using custom 1D CUDA kernels
Step4: Computing the final output Y = ΔW × X, where X is an input matrix


Implementation Methods:
1) Normal Matrix Multiplication: Standard dense computation of ΔW using full Hadamard and coefficient matrices.
2) Block-Diagonal Optimization (Planned): Decomposes ΔW into independent Hadamard blocks to reduce memory usage and improve parallelism.

Testing:
