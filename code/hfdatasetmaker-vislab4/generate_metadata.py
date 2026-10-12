# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
import json
import argparse
from tqdm import tqdm
from PIL import Image


def generate_metadata(image_folder, output_json, include_size=True):
    metadata = []

    image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, filename)
        entry = {"image_path": image_path, "caption": ""}

        if include_size:
            try:
                with Image.open(image_path) as img:
                    entry["width"], entry["height"] = img.size
            except Exception as e:
                print(f"Unable to process image {image_path}: {e}")
                entry["width"], entry["height"] = None, None

        metadata.append(entry)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"Metadata JSON saved: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata JSON for image dataset.")
    parser.add_argument("image_folder", type=str, help="Path to the image folder")
    parser.add_argument("output_json", type=str, help="Output JSON file path")
    parser.add_argument(
        "--include-size", action="store_true", help="Include image width and height"
    )

    args = parser.parse_args()

    generate_metadata(args.image_folder, args.output_json, args.include_size)
