import h5py
import numpy as np
import argparse
import sys
import os

def explore_h5_group(group, prefix=''):
    """Recursively explores and prints the structure of an HDF5 group or file."""
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            print(f"{prefix}Dataset: '{key}'")
            print(f"{prefix}  Shape: {item.shape}")
            print(f"{prefix}  Type:  {item.dtype}")
            # Example to show min/max for numeric data to understand value range (optional)
            if np.issubdtype(item.dtype, np.number) and item.size > 0:
                print(f"{prefix}  Range: [{np.min(item)}, {np.max(item)}]")
        elif isinstance(item, h5py.Group):
            print(f"{prefix}Group: '{key}'")
            explore_h5_group(item, prefix + '  ')

def read_h5_data(filepath):
    """
    Reads data from the specified HDF5 file.
    Lists its structure and loads root datasets into numpy arrays.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        sys.exit(1)

    print(f"Reading HDF5 file: {filepath}\n" + "="*50)
    
    try:
        with h5py.File(filepath, 'r') as f:
            print("--- HDF5 File Structure ---")
            explore_h5_group(f)
            print("-" * 27)

            # Loads datasets at the root into a dictionary of numpy arrays
            data_dict = {}
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    # Reading the data into memory as a numpy array
                    data_dict[key] = np.array(f[key])
                    print(f"\nSuccessfully loaded dataset '{key}' into memory.")
                    print(f"-> Numpy array shape: {data_dict[key].shape}, dtype: {data_dict[key].dtype}")
                    
            print("="*50)
            return data_dict
            
    except Exception as e:
        print(f"Failed to read HDF5 file '{filepath}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and explore an HDF5 file.")
    # Default file path relative to this script's location
    default_h5_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "AdobeIndoorNav",
        "data",
        "panorama_images_cropped_rgb_images",
        "et07-cr-galgary.h5"
    )
    
    parser.add_argument(
        "filepath", 
        type=str, 
        nargs="?", 
        default=default_h5_path,
        help="Path to the .h5 file to read"
    )
    
    args = parser.parse_args()
    
    # Run the read function
    data = read_h5_data(args.filepath)
    
    # Note: If the file contains image data, you could further process it here.
    # For example, if it has a dataset 'image' or shape (H, W, 3):
    # import matplotlib.pyplot as plt
    # plt.imshow(data['image'])
    # plt.show()
