# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
from datasets import load_dataset
import os

gdrive_root = os.getenv("DEEPGAZE_DATA_ROOT", "./data")

original_dir = os.path.join(gdrive_root, "mit_1003_image_original")
resized_dir = os.path.join(gdrive_root, "mit_1003_image_resized")

os.makedirs(original_dir, exist_ok=True)
os.makedirs(resized_dir, exist_ok=True)

# Load a Hugging Face dataset
dataset = load_dataset("qiuweihao/mit_1003_zeros_bias_landscapeonly_split", split="train")

# Save each image
for idx, item in enumerate(dataset):
    filename = os.path.basename(item["image_path"])

    # Save the original image
    item["image_original"].save(os.path.join(original_dir, filename))

    # Save the resized image
    item["image_resized"].save(os.path.join(resized_dir, filename))

    if idx % 100 == 0:
        print(f"Saved {idx} images")

print("All images have been saved!")
