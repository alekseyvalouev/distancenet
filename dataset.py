import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2

import os
import h5py

from tqdm import tqdm
import random
from start_end_only import get_all_valid_paths, get_traj_images, create_gif

class DistanceNetDataset(Dataset):
    def __init__(self, scenes=["et07-cr-galgary", "et12-kitchen"], horizons=[2, 4, 8, 16], negative_samples=True, transform=None, classification=True):
        self.scenes = scenes
        self.horizons = horizons
        self.negative_samples = negative_samples
        self.transform = transform
        self.classification = classification
        self.processed_data_folder = "/hdd/cached_adobeindoornav/"
        self.raw_data_folder = "/home/alekseyvalouev/goalnav/AdobeIndoorNav/datasets/adobeindoornav_dataset"
        self.scene_to_images = {}
        self.scene_to_grid = {}
        self._load_data()
        self.labels = {}
        for i in range(len(self.horizons)):
            self.labels[self.horizons[i]] = i
        if self.negative_samples:
            self._load_negative_samples()
            self.labels[-1] = len(self.horizons)

    def _load_data(self):
        self.data = []
        for scene_name in self.scenes:
            grid_id, _ = self._load_scene(scene_name)
            self.scene_to_grid[scene_name] = grid_id
            valid_nodes = grid_id[grid_id != -1]

            for horizon in self.horizons:
                trajectories = get_all_valid_paths(grid_id, len(valid_nodes), horizon, scene_name)
                self.data.extend(trajectories)
        
        print(f"Loaded dataset with {len(self.data)} samples.")
    
    def _load_scene(self, scene_name):
        occ_input_file = os.path.join(self.raw_data_folder, scene_name, 'grid_occ.npy')
        connect_input_file = os.path.join(self.raw_data_folder, scene_name, 'grid_connect.npy')

        with open(occ_input_file, 'rb') as fin:
            grid_id = np.load(fin)
            
        with open(connect_input_file, 'rb') as fin:
            grid_connect = np.load(fin)
        
        return grid_id, grid_connect
    
    def _make_random_sample(self, scene_name):
        grid_id = self.scene_to_grid[scene_name]
        valid_nodes = grid_id[grid_id != -1]
        start_id = random.choice(valid_nodes)
        angle = random.choice([0, 90, 180, 270])

        return (start_id, angle)
    
    def _load_negative_samples(self, p=0.25):
        # We generate len(data) * p negative samples. 
        n_negatives = int(len(self.data) * p)
        self.negative_samples = []
        for _ in range(n_negatives):
            scene_start, scene_end = np.random.choice(self.scenes, size=2, replace=False)
            start_pair = self._make_random_sample(scene_start)
            end_pair = self._make_random_sample(scene_end)
            self.negative_samples.append(((scene_start, scene_end), -1, (start_pair, end_pair)))
        
        print(f"Generated {len(self.negative_samples)} negative samples.")
        
        self.data.extend(self.negative_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scenes, horizon, pair = self.data[idx]

        start_img = self._get_image(scenes[0], pair[0])
        end_img = self._get_image(scenes[1], pair[1])

        if self.transform:
            start_img = self.transform(start_img)
            end_img = self.transform(end_img)
        
        label = horizon
        if self.classification:
            label = self.labels[horizon]
        if label == -1:
            label = max(self.horizons)

        return start_img, end_img, label
    
    def _get_image(self, scene, point):
        img_id = int(point[0]*4 + (point[1]//90))
        image_path = os.path.join(self.processed_data_folder, scene, f'{img_id}.png')
        return np.array(Image.open(image_path))
        

if __name__ == "__main__":
    dataset = DistanceNetDataset()
    start_img, end_img, horizon = dataset[0]
    print(start_img.shape, end_img.shape, horizon)
    create_gif([start_img, end_img], "test.gif")