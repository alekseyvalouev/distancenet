import h5py
import numpy as np
import sys
import os

filepath = "/home/alekseyvalouev/goalnav/AdobeIndoorNav/data/panorama_images_cropped_rgb_images/et07-cr-galgary.h5"

try:
    with h5py.File(filepath, 'r') as f:
        print("--- HDF5 File Structure ---")
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
        print(data_dict)
except Exception as e:
    print(f"Failed to read HDF5 file '{filepath}': {e}")
    sys.exit(1)