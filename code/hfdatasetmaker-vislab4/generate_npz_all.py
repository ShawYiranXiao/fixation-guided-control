# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
from PIL import Image
from processors.deepgaze_feature import deepgaze_process

# === Configure paths ===
root_dir = "dataset/GeorgeCurationRenamedOriginal"
output_root = os.getenv("DEEPGAZE_NPZ_ROOT", "./outputs/npz")

centerbias_modes = {"original": "original", "zeros": "zeros"}

subfolders = ["Complex", "NonComplex"]

# === Main workflow ===
for centerbias_key, centerbias_type in centerbias_modes.items():
    for subfolder in subfolders:
        input_dir = os.path.join(root_dir, subfolder)
        output_dir = os.path.join(output_root, centerbias_key, subfolder)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n Processing: {input_dir} | Centerbias: {centerbias_type}")

        for filename in sorted(os.listdir(input_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(input_dir, filename)
            try:
                image = Image.open(image_path).convert("RGB")
                print(f"▶ Processing {filename}...")

                batch = {"image_original": [image], "key": [os.path.splitext(filename)[0]]}

                # process_image_with_deepgaze_batch(
                #     image=image,
                #     num_points=4,
                #     batch_size=1,
                #     total_iterations=10,
                #     centerbias_type=centerbias_type,
                #     original_image_path=image_path,
                #     npz_output_dir=output_dir
                # )
                params = {
                    "num_points": 4,
                    "batch_random_size": 1,
                    "total_iterations": 10,
                    "centerbias": centerbias_type,
                    "feature_method": "mean",
                    "npz_output_dir": output_dir,
                    "image_paths": [image_path],
                }

                result = deepgaze_process(
                    batch, input_key="image_original", output_key="deepgaze_feature", params=params
                )

                save_path = os.path.join(
                    output_dir, f"{os.path.splitext(filename)[0]}_colormap.jpg"
                )
                result["deepgaze_feature"][0].save(save_path)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

print("\n All Done!")
