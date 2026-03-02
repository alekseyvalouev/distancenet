import os 
import shutil

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

for scene_name in scene_names:
    occ_input_file = os.path.join("/home/alekseyvalouev/goalnav/AdobeIndoorNav/datasets/adobeindoornav_dataset", scene_name, 'grid_occ.npy')
    connect_input_file = os.path.join("/home/alekseyvalouev/goalnav/AdobeIndoorNav/datasets/adobeindoornav_dataset", scene_name, 'grid_connect.npy')
    
    occ_output_file = os.path.join("/hdd/cached_adobeindoornav", scene_name, 'grid_occ.npy')
    connect_output_file = os.path.join("/hdd/cached_adobeindoornav", scene_name, 'grid_connect.npy')
    
    os.makedirs(os.path.dirname(occ_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(connect_output_file), exist_ok=True)
    
    shutil.copy(occ_input_file, occ_output_file)
    shutil.copy(connect_input_file, connect_output_file)