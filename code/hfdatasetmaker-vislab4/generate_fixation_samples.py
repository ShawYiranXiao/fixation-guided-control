# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
import sys
from PIL import Image
import numpy as np

# Make the local modules importable
sys.path.append(os.path.abspath("."))

from processors.deepgaze_feature import (
    deepgaze_model,
    centerbias_template_original,
    centerbias_template_zeros,
)
from processors.deepgaze_feature import process_image_with_deepgaze_batch, apply_colormap

# Input image path
input_image_path = "C-streetPhotography-5.jpg"  # or a path to another input image
image = Image.open(input_image_path).convert("RGB")

# Configure parameters
dataset_key = "c_streetphoto_5"
npz_output_dir = "fixation_outputs"
num_fixations = 4
batch_size = 1
total_iterations = 10
feature_method = "mean"
centerbias_type = "original"  # or "zeros"

# Choose the center-bias template
centerbias_template = (
    centerbias_template_zeros if centerbias_type == "zeros" else centerbias_template_original
)

# Generate heatmaps and save them as NPZ
_ = process_image_with_deepgaze_batch(
    model=deepgaze_model,
    centerbias_template=centerbias_template,
    image=image,
    num_points=num_fixations,
    batch_size=batch_size,
    total_iterations=total_iterations,
    dataset_key=dataset_key,
    npz_output_dir=npz_output_dir,
    feature_method=feature_method,
)

# Load cached heatmaps
npz_path = os.path.join(npz_output_dir, f"{dataset_key}_heatmaps.npz")
heatmaps = np.load(npz_path)["heatmaps"]  # shape: (10, H, W)

# Visualize and save color heatmaps
output_image_dir = os.path.join(npz_output_dir, "individual_maps")
os.makedirs(output_image_dir, exist_ok=True)

for i, heatmap in enumerate(heatmaps):
    colored = apply_colormap(heatmap)
    out_path = os.path.join(output_image_dir, f"{dataset_key}_sample_{i}.jpg")
    colored.save(out_path)
    print(f"Saved: {out_path}")
