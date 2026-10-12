# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
import csv
import numpy as np
import torch
from PIL import Image as PILImage
from scipy.ndimage import zoom
from scipy.special import logsumexp
from deepgaze_pytorch import DeepGazeIII


def simulate_scanpath(
    image_path,
    output_csv_path="simulated_scanpath.csv",
    max_duration=20.0,
    min_fixation=0.03,
    max_fixation=0.08,
    seed=42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepGazeIII(pretrained=True).to(device).eval()

    # Load image
    image = PILImage.open(image_path).convert("RGB")
    image_np = np.array(image)
    H, W = image_np.shape[:2]

    # Load and resize centerbias
    centerbias_template = np.load(os.getenv("DEEPGAZE_CENTERBIAS_PATH", "centerbias_mit1003.npy"))
    cb = zoom(
        centerbias_template,
        (H / centerbias_template.shape[0], W / centerbias_template.shape[1]),
        order=1,
    )
    cb -= logsumexp(cb)
    centerbias_tensor = torch.tensor([cb], dtype=torch.float32).to(device)

    # Image tensor
    img_tensor = torch.tensor([image_np.transpose(2, 0, 1)], dtype=torch.float32).to(device)

    # Fixation history starts from center
    x_hist = [W // 2] * 4
    y_hist = [H // 2] * 4
    scanpath = []
    timestamp = 0.0

    with torch.no_grad():
        while timestamp < max_duration:
            x_tensor = torch.tensor([x_hist[-4:]], dtype=torch.float32).to(device)
            y_tensor = torch.tensor([y_hist[-4:]], dtype=torch.float32).to(device)

            log_density = model(img_tensor, centerbias_tensor, x_tensor, y_tensor)
            heatmap = log_density.exp().cpu().numpy()[0, 0]

            flat = heatmap.flatten()
            flat /= flat.sum()
            idx = np.random.choice(len(flat), p=flat)
            y_px, x_px = divmod(idx, W)

            # Convert to normalized coordinates
            x_norm = x_px / W
            y_norm = y_px / H

            scanpath.append([round(timestamp, 4), x_norm, y_norm])

            dt = float(np.clip(np.random.normal(loc=0.05, scale=0.01), min_fixation, max_fixation))
            timestamp += dt

            x_hist.append(x_px)
            y_hist.append(y_px)

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y"])
        writer.writerows(scanpath)

    print(f"Simulated scanpath saved to {output_csv_path} with {len(scanpath)} points.")


#  Example run
simulate_scanpath("01_baldessari.jpeg", "simulated_scanpath.csv")
