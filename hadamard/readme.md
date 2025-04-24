# GPU-Accelerated Matrix Operations with Hadamard Transformations (CUDA)

This part of the course project implements GPU-accelerated matrix operations using Hadamard transformations in CUDA.

## Computation Flow

1. **Construct Sparse Matrix `C`**  
   Load values and locations from `.npy` files and build the matrix.

2. **Generate Hadamard Matrices `H` and `Hᵗ`**  
   Create orthogonal matrices used for transformation.

3. **Calculate ΔW = H × C × Hᵗ**  
   Use custom 1D CUDA kernels for efficient matrix multiplication.

4. **Compute Final Output Y = ΔW × X**  
   Multiply the result with the input matrix `X`.

---

## Implementation Methods

1. **Normal Matrix Multiplication**  
   Standard dense computation of ΔW using full Hadamard and coefficient matrices.

2. **Block-Diagonal Optimization**  
   Decomposes ΔW into independent Hadamard blocks to reduce memory usage and improve parallelism.

---

## Testing Experiments

### `experiment1_hadamardx.sh`

- **Model:** LLaMA-2-7B  
- **Methods:** Method1, Method2  
- **Tokens:** 1024  
- **Threads per Block:** 32, 128, 512, 1024

### `experiment2_hadamardx.sh`

- **Models:** LLaMA-2-7B, LLaMA-2-13B, Gemma-2-2B, Gemma-2-9B, LLaMA-3.1-8B  
- **Methods:** Method1, Method2 for Query and Value layers  
- **Tokens:** 1024  
- **Threads per Block:** 1024

### `experiment3_hadamardx.sh`

- **Model:** LLaMA-2-7B  
- **Methods:** Method1, Method2  
- **Tokens:** 1, 16, 128, 512, 1024  
- **Threads per Block:** 1024

---

## Notes

- Implemented in CUDA with custom matrix multiplication kernels.
- Uses `cnpy` for `.npy` file parsing.
- Optimized for memory and performance on large-scale transformer models.
