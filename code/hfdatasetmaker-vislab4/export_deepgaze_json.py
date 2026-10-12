# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
import json
import numpy as np
from PIL import Image as PILImage
import torch
from deepgaze_pytorch.deepgaze3 import DeepGazeIII
from scipy.ndimage import zoom
from scipy.special import logsumexp


def extract_topk_fixations(heatmap, k=15):
    H, W = heatmap.shape
    flat_indices = np.argpartition(heatmap.flatten(), -k)[-k:]
    sorted_indices = flat_indices[np.argsort(-heatmap.flatten()[flat_indices])]
    fixations = []
    for idx in sorted_indices:
        y, x = divmod(idx, W)
        fixations.append({"x": int(x), "y": int(y), "value": float(heatmap[y, x])})
    return fixations


def export_fixations_to_json(image_path, output_json_path, dataset_key="image_001"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    deepgaze_model = DeepGazeIII(pretrained=True).to(device).eval()

    centerbias_path = os.path.join("deepgaze_pytorch", "centerbias_mit1003.npy")
    centerbias_template = np.load(centerbias_path)

    image = PILImage.open(image_path).convert("RGB")
    image_np = np.array(image)
    H, W = image_np.shape[:2]

    # Resize and normalize centerbias
    cb = zoom(
        centerbias_template,
        (H / centerbias_template.shape[0], W / centerbias_template.shape[1]),
        order=1,
    )
    cb -= logsumexp(cb)
    centerbias_tensor = torch.tensor([cb], dtype=torch.float32).to(device)

    # Convert image to tensor
    img_tensor = torch.tensor([image_np.transpose(2, 0, 1)], dtype=torch.float32).to(device)

    # Fixed center point as fixation history
    x_hist = torch.tensor([[W // 2] * 4], dtype=torch.float32).to(device)
    y_hist = torch.tensor([[H // 2] * 4], dtype=torch.float32).to(device)

    # Predict saliency
    with torch.no_grad():
        log_density = deepgaze_model(img_tensor, centerbias_tensor, x_hist, y_hist)
        heatmap = log_density.exp().cpu().numpy()[0, 0]

    norm_heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    fixations = extract_topk_fixations(norm_heatmap, k=15)

    json_data = {"key": dataset_key, "width": W, "height": H, "fixations": fixations}

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Fixations saved to: {output_json_path}")


if __name__ == "__main__":
    export_fixations_to_json(
        image_path="01_baldessari.jpeg",
        output_json_path="output/fixations_image_001.json",
        dataset_key="image_001",
    )
