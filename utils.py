import os
import pandas as pd
import yaml
from typing import Literal
import json
import torch
from datasets import load_dataset

def load_from_yaml(path_to_file: str):
    print(f"loading data from .yaml file @ {path_to_file}")
    with open(path_to_file) as file:
        _dict = yaml.safe_load(file)
    return _dict

def load_from_txt(path_to_file: str):
    print(f"loading data from .txt file @ {path_to_file}")
    with open (path_to_file, "r") as myfile:
        data = myfile.read().splitlines()
    return data

def save_to_json(path_to_file: str, data: list):
    with open(path_to_file, 'w') as outfile:
        json.dump(data, outfile)
    print(f"file saved @ loc: {path_to_file}")

def load_from_json(path_to_file: str):
    print(f"loading data from .json file @ {path_to_file}")
    with open(path_to_file, "r") as json_file:
        _dict = json.load(json_file)
    return _dict

_split_types = Literal["train", "test", "valid"]
def load_to_dataframe(path_to_file: str, split: _split_types):
    """
        path_to_file: .json file
        Note: utilizes datasets.load_dataset module to load raw data
    """
    print(f"loading data from .json file @ {path_to_file} using datasets.load_dataset module")
    return pd.DataFrame(load_dataset("json", data_files=path_to_file, split="train"))

def check_and_create_directory(path_to_folder):
    """
    check if a nested path exists and create 
    missing nodes/directories along the route
    """
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    return path_to_folder

path_to_config_file = "./config.yaml"
config = load_from_yaml(path_to_file=path_to_config_file)

check_and_create_directory(config["PATH_TO_RESULT_OUTPUT_DIR"])
check_and_create_directory(config["PATH_TO_MODEL_OUTPUT_DIR"])

os.makedirs(config["PATH_TO_RESULT_OUTPUT_DIR"] + "./val/", exist_ok=True)
os.makedirs(config["PATH_TO_RESULT_OUTPUT_DIR"] + "./test/", exist_ok=True)

config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"