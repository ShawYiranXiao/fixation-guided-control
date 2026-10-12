# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
from datasets import Dataset, DatasetDict, Features, Value, Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image as PILImage
import torch

# Hugging Face account
HF_USERNAME = os.getenv("HF_USERNAME")
REPO_NAME = "MIT1003-randomattention"
repo_id = f"{HF_USERNAME}/{REPO_NAME}"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(
    device
)


# Generate a caption
def generate_caption(image_path):
    image = PILImage.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    caption_ids = model.generate(**inputs)
    caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
    return caption


# Parse the data
def scan_dataset(root_dir):
    dataset_splits = {}

    for setname in os.listdir(root_dir):
        set_path = os.path.join(root_dir, setname)
        if not os.path.isdir(set_path):
            continue  # Process directories only

        data = []
        for example_name in os.listdir(set_path):
            example_path = os.path.join(set_path, example_name)
            if not os.path.isdir(example_path):
                continue

            final_feature_map_path = os.path.join(example_path, "final_feature_map.png")
            original_image_path = os.path.join(example_path, "original_image.jpg")

            if os.path.exists(final_feature_map_path) and os.path.exists(original_image_path):
                caption = generate_caption(original_image_path)  # Generate a caption

                # **Load and store image objects instead of paths**
                condition_image = PILImage.open(final_feature_map_path).convert("RGB")
                original_image = PILImage.open(original_image_path).convert("RGB")

                data.append(
                    {
                        "example_name": example_name,
                        "condition": condition_image,  #  Store image data directly
                        "original": original_image,  #  Store image data directly
                        "caption": caption,  # Generate a caption
                    }
                )
            else:
                print(f"Warning: Missing files in {example_path}")

        # Include only nonempty dataset splits
        if data:
            dataset_splits[setname] = Dataset.from_list(data)

    return dataset_splits


# Process data
dataset_splits = scan_dataset("./MIT1003_10random_features")

# Declare image columns as datasets.Image features
features = Features(
    {
        "example_name": Value("string"),
        "condition": Image(decode=True),  #  Store image data directly
        "original": Image(decode=True),  #  Store image data directly
        "caption": Value("string"),
    }
)

# Create a DatasetDict
dataset_dict = DatasetDict(
    {split_name: dataset.cast(features) for split_name, dataset in dataset_splits.items()}
)

# **Remove existing output before rebuilding the dataset**
os.system("rm -rf huggingface_dataset")

dataset_dict.save_to_disk("huggingface_dataset")

# **Upload to Hugging Face**
dataset_dict.push_to_hub(repo_id)
print(f"Dataset successfully updated: https://huggingface.co/datasets/{repo_id}")
