# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import argparse
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# === Argument parsing ===
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", choices=["gray", "colormap"], default="gray", help="Comparison mode: gray or colormap"
)
args = parser.parse_args()

# === Display mode ===
mode = args.mode
base_dir = f"./outputs/fusion_comparison_{mode}"
output_dir = f"./outputs/comparisons_{mode}"
os.makedirs(output_dir, exist_ok=True)

# === Image comparison configuration ===
fusion_methods = ["softor", "fusion_softmax_sum", "fusion_softmax_lighten"]
bias_types = ["original", "zeros"]
groups = ["Complex", "NonComplex"]
num_samples = 6

grids_info = {}

for method in fusion_methods:
    grids_info[method] = []
    for group in groups:
        for bias in bias_types:
            folder = os.path.join(base_dir, bias, group)
            if not os.path.exists(folder):
                print(f"Path does not exist: {folder}")
                grids_info[method].append([])
                continue
            all_images = [
                f
                for f in os.listdir(folder)
                if f.endswith(f"{method}.jpg") and not f.startswith("._")
            ]
            selected = random.sample(all_images, min(num_samples, len(all_images)))
            grids_info[method].append([os.path.join(folder, f) for f in selected])

# === Plot output ===
for method, columns in grids_info.items():
    fig, axes = plt.subplots(
        num_samples, len(columns), figsize=(4 * len(columns), 3.5 * num_samples)
    )
    fig.suptitle(f"{mode.upper()} | Fusion Method: {method}", fontsize=18)

    for col_idx, col in enumerate(columns):
        for row_idx, img_path in enumerate(col):
            ax = axes[row_idx][col_idx] if num_samples > 1 else axes[col_idx]
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"{groups[col_idx // 2]} / {bias_types[col_idx % 2]}", fontsize=10)
            ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(output_dir, f"comparison_{method}.jpg")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
