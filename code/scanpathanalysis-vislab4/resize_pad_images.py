# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
import sys
from PIL import Image, ImageOps


def resize_and_pad(image_path, output_path, target_size=(1024, 768)):
    image = Image.open(image_path)
    image_ratio = image.width / image.height
    target_ratio = target_size[0] / target_size[1]

    if image_ratio > target_ratio:
        new_width = target_size[0]
        new_height = round(target_size[0] / image_ratio)
    else:
        new_height = target_size[1]
        new_width = round(target_size[1] * image_ratio)

    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    background = Image.new("RGB", target_size, (0, 0, 0))
    paste_position = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)
    background.paste(resized, paste_position)
    background.save(output_path)


def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            resize_and_pad(input_path, output_path)
            print(f"Processed: {filename}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python resize_images.py <input_folder> <output_folder>")
        sys.exit(1)
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    process_folder(input_folder, output_folder)
