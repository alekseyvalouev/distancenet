# take all images in every scne. write them to cached/scene_name/id.png
import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm

scene_names = [
    "et07-imagination-lab",
    "et12-corner-2",
    "et12-cr-helsinki",
    "et12-cr-honolulu",
    "et12-cr-hamburg",
    "et12-office-104",
    "et12-office-108",
    "et12-office-110",
    "et12-office-111",
    "et12-office-112",
    "et12-office-113",
    "et12-office-114",
    "et12-office-115",
    "et12-office-117",
    "et12-office-132",
    "et12-corner-1",
    "et07-cr-galgary",
    "et12-cr-hongkong",
    "et12-kitchen",
    "et07-office-114",
    "et07-office-419",
    "et07-office-420",
    "et07-office-423",
    "et07-office-424",
]

for scene_name in tqdm(scene_names):

    processed_data_folder = "/home/alekseyvalouev/goalnav/AdobeIndoorNav/data"
    h5_path = os.path.join(processed_data_folder, 'panorama_images_cropped_rgb_images', f'{scene_name}.h5')
    h5_file = h5py.File(h5_path, 'r')
    images = h5_file['observation']

    out_dir = f'/hdd/cached_adobeindoornav/{scene_name}'
    os.makedirs(out_dir, exist_ok=True)

    for i, image in tqdm(enumerate(images), total=len(images), desc=scene_name):
        image = np.array(image).reshape(512, 512, 3)
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(os.path.join(out_dir, f'{i}.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))