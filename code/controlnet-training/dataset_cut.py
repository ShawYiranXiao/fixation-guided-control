# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
import shutil
import random


def split_folders(source_dir, train_dir, val_dir, split_ratio=0.8):
    """
    Split source subdirectories between train_dir and val_dir using the requested ratio
    """
    # Ensure the destination directory exists
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # List all subdirectories
    subfolders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

    # Shuffle records
    random.shuffle(subfolders)

    # Split by the requested ratio
    split_point = int(len(subfolders) * split_ratio)
    train_subfolders = subfolders[:split_point]
    val_subfolders = subfolders[split_point:]

    # Move the directory
    for folder in train_subfolders:
        shutil.move(os.path.join(source_dir, folder), os.path.join(train_dir, folder))

    for folder in val_subfolders:
        shutil.move(os.path.join(source_dir, folder), os.path.join(val_dir, folder))

    print(f"Train folders: {len(train_subfolders)}, Validation folders: {len(val_subfolders)}")


# Example usage
source_directory = "ALLSTIMULI"
train_directory = "MIT1003_10random_features/train"
validation_directory = "MIT1003_10random_features/validation"

split_folders(source_directory, train_directory, validation_directory, split_ratio=0.9)
