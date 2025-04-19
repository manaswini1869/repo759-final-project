import numpy as np
import os

def convert_npy_to_bin(npy_file, bin_file):
    data = np.load(npy_file)
    data = data.astype(np.float32)  # ensure float32 for C++ compatibility
    data.tofile(bin_file)
    print(f"Saved {bin_file}")

if __name__ == "__main__":
    # List of your input files (only filenames)
    files_to_convert = [
        ("gemma-2-2b-hadamard-CT.npy", "C_values.bin"),     # C values
        ("gemma-2-2b-hadamard-locs.npy", "C_locs.bin"),      # C locs
        ("x_1.npy", "x_1.bin"),
        ("x_16.npy", "x_16.bin"),
        ("x_128.npy", "x_128.bin"),
        ("x_512.npy", "x_512.bin"),
        ("x_1024.npy", "x_1024.bin"),
    ]

    # Source folder where the .npy files are
    input_folder = "inputs_npy"

    # Destination folder to save .bin files
    output_folder = "inputs_bin"   # same folder, or you can create a new one if needed

    os.makedirs(output_folder, exist_ok=True)

    for npy_file, bin_file in files_to_convert:
        npy_path = os.path.join(input_folder, npy_file)
        bin_path = os.path.join(output_folder, bin_file)
        convert_npy_to_bin(npy_path, bin_path)
