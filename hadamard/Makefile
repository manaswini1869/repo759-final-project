// ==========================================
// Makefile
// ==========================================
# Simple Makefile

TARGET = run
NVCC = /usr/local/cuda/bin/nvcc

SRC = main.cu hadamard.cpp matrix_ops.cpp utils.cpp

all:
	$(NVCC) $(SRC) -lcublas -o $(TARGET)

clean:
	rm -f $(TARGET)