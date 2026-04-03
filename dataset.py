import torch
import torchvision
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
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

class SacsonDataset(Dataset):
    def __init__(self, scenes, horizons=[2, 4, 8, 16], context_size=0, negative_samples=True, transform=None, classification=True):
        # We assume the mean distance between samples is 0.2 meters. 
        # This means that the label is delta(indices) * 0.4
        self.scale_factor = 0.4
        self.horizons = []
        self.labels = {}
        # for our output, we do [B, C*(context_size + 1), H, W]
        self.context_size = context_size
        for i, h in enumerate(horizons):
            assert h / self.scale_factor % 1 == 0, "Horizon must be a multiple of 0.4 (scaling factor for Sacson)"
            self.horizons.append(int(h / self.scale_factor))
            self.labels[int(h / self.scale_factor)] = i
        self.scenes = scenes
        self.negative_samples = negative_samples
        self.transform = transform
        self.classification = classification
        self._load_data()
        if self.negative_samples:
            self._load_negative_samples()
            self.labels[-1] = -1
    
    def _load_data(self):
        self.data = []
        self.data_path = "/hdd/sacson"
        for scene in self.scenes:
            folder = os.path.join(self.data_path, scene)
            # we're in the folder. get the highest #'ed image. 
            images = list(filter(lambda x: x.endswith(".jpg"), os.listdir(folder)))
            images.sort(key=lambda x: int(x.split(".")[0]))
            highest_image = images[-1]
            max_idx = int(highest_image.split(".")[0])

            for h in self.horizons:
                for i in range(max_idx - h + 1):
                    start_idx = i
                    end_idx = i + h
                    start_img = images[start_idx]
                    start_idx = int(start_img.split(".")[0])
                    if start_idx < self.context_size:
                        continue
                    end_img = images[end_idx]
                    start_img = os.path.join(folder, start_img)
                    end_img = os.path.join(folder, end_img)
                    self.data.append((start_img, end_img, h))

        print(f"Loaded dataset with {len(self.data)} samples.")
    
    def _make_random_sample(self, scene, min_idx=0):
        folder = os.path.join(self.data_path, scene)
        images = list(filter(lambda x: x.endswith(".jpg") and int(x.split(".")[0]) >= min_idx, os.listdir(folder)))
        image = random.choice(images)
        return os.path.join(folder, image)

    def _load_negative_samples(self):
        self.negative_samples = []
        p = 0.5
        for _ in range(int(len(self.data) * p)):
            while True:
                scene_start, scene_end = np.random.choice(self.scenes, size=2, replace=False)
                try:
                    start_img = self._make_random_sample(scene_start, min_idx=self.context_size)
                    end_img = self._make_random_sample(scene_end)
                    self.negative_samples.append((start_img, end_img, -1))
                    break
                except Exception as e:
                    continue
            
        self.data.extend(self.negative_samples)

        print(f"Generated {len(self.negative_samples)} negative samples.")
    
    def _get_image(self, image_path):
        img = Image.open(image_path).convert("RGB")

        # Resize so that the smallest dimension is 224 while preserving aspect ratio
        width, height = img.size
        scale = 224.0 / min(width, height)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        img = img.resize((new_width, new_height), Image.BILINEAR)

        # Center crop to 224x224
        left = (new_width - 224) // 2
        top = (new_height - 224) // 2
        right = left + 224
        bottom = top + 224
        img = img.crop((left, top, right, bottom))

        return np.array(img)
    
    def _get_all(self, *paths):
        return np.concatenate([self._get_image(path) for path in paths], axis=2)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        start_img, end_img, horizon = self.data[idx]
        # basically we want start_img and the previous context_size images before it. 
        start_idx = int(start_img.split("/")[-1].split(".")[0])
        end_idx = int(end_img.split("/")[-1].split(".")[0])
        start_name, end_name = start_img.split("/")[-1], end_img.split("/")[-1]
        start_scene, end_scene = start_img.split("/")[-2], end_img.split("/")[-2]

        folder = os.path.join(self.data_path, start_scene)
        start_imgs = []
        for i in range(start_idx - self.context_size, start_idx):
            start_imgs.append(os.path.join(folder, f"{i}.jpg"))
        start_imgs.append(start_img)
        try:
            start_imgs = self._get_all(*start_imgs)
        except Exception as e:
            print(e)
            print(start_imgs)
            raise RuntimeError("Failed to get all start images")
        end_img = self._get_image(end_img)
        if self.transform:
            start_imgs = self.transform(start_imgs)
            end_img = self.transform(end_img)
        label = horizon
        #if label == -1 and not self.classification:
        #    label = max(self.horizons)
        if self.classification:
            label = self.labels[horizon]
        else:
            if label != -1:
                label = int(label*self.scale_factor)
            else:
                label = max(self.horizons)*self.scale_factor*2

        #return start_name, end_name, start_scene, end_scene, start_imgs, end_img, label
        return start_imgs, end_img, label

def make_mix_dataset(adobe_scenes, sacson_scenes, adobe_horizons=[2, 4, 8, 16], sacson_horizons=[2, 4, 8, 16], negative_samples=True, adobe_transform=None, sacson_transform=None, classification=False):
    adobe_dataset = DistanceNetDataset(scenes=adobe_scenes, horizons=adobe_horizons, negative_samples=negative_samples, transform=adobe_transform, classification=classification)
    sacson_dataset = SacsonDataset(scenes=sacson_scenes, horizons=sacson_horizons, negative_samples=negative_samples, transform=sacson_transform, classification=classification)
    return ConcatDataset([adobe_dataset, sacson_dataset])
            

if __name__ == "__main__":
    sacson_scenes = [
        "Dec-15-2022-bww8_00000000_0",
        "Jan-17-2023-bww8_00000001_5",
        "Feb-23-2023-soda3-intloss_00000000_1",
    ]
    adobe_scenes = [
        "et07-imagination-lab",
        "et12-corner-2",
        "et12-cr-helsinki",
        "et12-cr-honolulu",
        "et12-cr-hamburg",
        "et12-office-104",
        "et12-office-108",
        "et12-office-110",
        "et12-office-111",
    ]
    #dataset = make_mix_dataset(adobe_scenes, sacson_scenes, adobe_horizons=[2, 4, 8, 16], sacson_horizons=[2, 4, 8, 16], negative_samples=True, adobe_transform=None, sacson_transform=None, classification=False)

    dataset = SacsonDataset(sacson_scenes, horizons=[2, 4, 8, 16], context_size=5, negative_samples=True, transform=transforms.ToTensor(), classification=False)

    label = 2
    i = 0

    while label in [2, 4, 8, 16]:
        start_name, end_name, start_scene, end_scene, start_img, end_img, label = dataset[i]
        i += 1
    print(start_img.shape, end_img.shape, label)
    #create_gif([start_img, end_img], "test.gif")