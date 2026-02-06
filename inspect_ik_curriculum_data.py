#!/usr/bin/env python3
"""Script to inspect ik_curriculum_data.npz file."""

import numpy as np
import sys
import os

def inspect_npz(npz_path):
    """Inspect the contents of an NPZ file."""
    if not os.path.exists(npz_path):
        print(f"Error: File '{npz_path}' not found!")
        return
    
    print(f"=" * 80)
    print(f"Inspecting: {npz_path}")
    print(f"=" * 80)
    print()
    
    # Load the NPZ file
    data = np.load(npz_path)
    
    # Get all keys
    keys = list(data.keys())
    print(f"Keys in NPZ file: {keys}")
    print(f"Total number of arrays: {len(keys)}")
    print(data["qpos"][0])
    print(data["qpos"][1])
    print(data["qpos"][49])
    
    # Inspect each array
    # for key in keys:
    #     array = data[key]
    #     print(f"-" * 80)
    #     print(f"Key: '{key}'")
    #     print(f"  Shape: {array.shape}")
    #     print(f"  Dtype: {array.dtype}")
    #     print(f"  Size: {array.size:,} elements")
    #     print(f"  Memory: {array.nbytes / 1024 / 1024:.2f} MB")
        
    #     # Statistics for numeric arrays
    #     if np.issubdtype(array.dtype, np.number):
    #         print(f"  Statistics:")
    #         print(f"    Min:    {np.min(array):.6f}")
    #         print(f"    Max:    {np.max(array):.6f}")
    #         print(f"    Mean:   {np.mean(array):.6f}")
    #         print(f"    Std:    {np.std(array):.6f}")
            
    #         # If 2D, show per-dimension stats
    #         if array.ndim == 2:
    #             print(f"  Per-dimension statistics (first 10 dimensions):")
    #             num_dims = min(10, array.shape[1])
    #             for i in range(num_dims):
    #                 dim_data = array[:, i]
    #                 print(f"    Dim {i:2d}: min={np.min(dim_data):8.4f}, max={np.max(dim_data):8.4f}, "
    #                       f"mean={np.mean(dim_data):8.4f}, std={np.std(dim_data):8.4f}")
    #             if array.shape[1] > 10:
    #                 print(f"    ... (showing first 10 of {array.shape[1]} dimensions)")
        
    #     # Show sample data
    #     print(f"  Sample data (first 3 rows):")
    #     if array.ndim == 1:
    #         print(f"    {array[:3]}")
    #     elif array.ndim == 2:
    #         print(f"    {array[:3, :]}")
    #     else:
    #         print(f"    {array[:3, ...]}")
        
    #     print()
    
    # # Summary
    # print(f"=" * 80)
    # print("Summary:")
    # total_size = sum(data[key].nbytes for key in keys)
    # print(f"  Total file size: {total_size / 1024 / 1024:.2f} MB")
    # print(f"  Number of arrays: {len(keys)}")
    # print(f"=" * 80)

if __name__ == "__main__":
    # Default file path
    npz_path = "ik_curriculum_data_merged.npz"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
    
    inspect_npz(npz_path)
