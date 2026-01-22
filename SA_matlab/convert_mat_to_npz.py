#!/usr/bin/env python3
"""
Convert MATLAB .mat files to NumPy .npz format.

Usage:
    python convert_mat_to_npz.py <input_directory> <output_directory>

Example:
    python convert_mat_to_npz.py ./matlab_data ./npz_data
"""

import os
import sys
import glob

import scipy.io as sio
import numpy as np


def convert_mat_to_npz(mat_file: str, output_dir: str) -> bool:
    """Convert a single .mat file to .npz format."""
    try:
        data = sio.loadmat(mat_file)
        save_dict = {k: v for k, v in data.items() if not k.startswith('__')}
        
        basename = os.path.basename(mat_file)
        name_without_ext = os.path.splitext(basename)[0]
        npz_filename = f"{name_without_ext}.npz"
        npz_path = os.path.join(output_dir, npz_filename)
        
        np.savez(npz_path, **save_dict)
        return True
    except Exception as e:
        print(f"Error converting {mat_file}: {e}")
        return False


def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_mat_to_npz.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Validate input directory
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .mat files
    mat_files = sorted(glob.glob(os.path.join(input_dir, "*.mat")))
    
    if not mat_files:
        print(f"No .mat files found in {input_dir}")
        sys.exit(0)
    
    # Convert all files
    success_count = 0
    for mat_file in mat_files:
        if convert_mat_to_npz(mat_file, output_dir):
            success_count += 1
    
    print(f"Successfully converted {success_count}/{len(mat_files)} files to {output_dir}")


if __name__ == '__main__':
    main()
