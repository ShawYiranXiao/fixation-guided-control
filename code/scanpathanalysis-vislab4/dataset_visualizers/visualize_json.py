# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import json
import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image


def visualize_fixations(json_path, image_path=None, save_path=None):
    # Read fixations
    with open(json_path, "r") as f:
        fixations = json.load(f)

    # Optional background image
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path)
        width, height = img.size
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.imshow(img, origin="upper")
    else:
        # Set the canvas size automatically
        width = 1024
        height = 768
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_facecolor("white")

    # Visualize fixation locations
    for fix in fixations:
        x_norm = fix["x"]
        y_norm = fix["y"]
        dur = fix["duration"]
        # Convert normalized coordinates to pixels
        x = x_norm * width
        y = y_norm * height
        # Scale the marker by fixation duration
        radius = dur * 100  # Adjust this scale factor as needed
        circle = plt.Circle((x, y), radius, facecolor="red", edgecolor="black", alpha=0.4)
        ax.add_patch(circle)
        ax.plot(x, y, "ro", markersize=4)

    ax.set_title("Fixations (size = duration)")
    ax.axis("off")

    # Save or display the result
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to fixation JSON file")
    parser.add_argument("--image", help="Optional background image")
    parser.add_argument("--out", help="Optional path to save output image")
    args = parser.parse_args()

    visualize_fixations(args.json, image_path=args.image, save_path=args.out)
