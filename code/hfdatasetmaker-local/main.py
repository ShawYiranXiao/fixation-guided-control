# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
from datasets import load_dataset, Image
from process_pipeline import process_dataset
from upload_dataset import load_local_dataset, upload_to_huggingface

#  1. Choose the processing stages
from processors.load_image import load_image
from processors.resize_image import resize_image
from processors.generate_caption import generate_caption
from processors.encode_image import encode_image
from processors.deepgaze_feature import deepgaze_process

processors = [
    {"processor_function": load_image, "num_proc": 8},
    {"processor_function": generate_caption, "num_proc": 1},
    {"processor_function": deepgaze_process, "num_proc": 1},
    {"processor_function": encode_image, "num_proc": 8},
]

dataset_path = "processed_dataset"

#  Read the Hugging Face token from the environment
HF_TOKEN = os.getenv("HF_TOKEN")  # Read from the environment
HF_USERNAME = os.getenv("HF_USERNAME")  #  Set your own Hugging Face username
HF_REPO_NAME = os.getenv("HF_REPO_NAME")  #  Set your own dataset repository name


if os.path.exists(dataset_path):
    raise FileExistsError(f"Output already exists: {dataset_path}. Choose a new directory.")

os.makedirs(dataset_path, exist_ok=True)

dataset = load_dataset("json", data_files="dataset/complex_images.json", split="train")
print(dataset)

dataset = process_dataset(dataset, processors, batch_size=5, load_cache=False)

dataset = dataset.cast_column("image", Image())
dataset = dataset.cast_column("deepgaze_feature", Image())

dataset.save_to_disk(dataset_path)

print("The dataset is ready for upload")

dataset_test = load_local_dataset(dataset_path)

if os.getenv("HF_UPLOAD", "0") == "1":
    upload_to_huggingface(dataset_test, HF_USERNAME, HF_REPO_NAME, HF_TOKEN)
