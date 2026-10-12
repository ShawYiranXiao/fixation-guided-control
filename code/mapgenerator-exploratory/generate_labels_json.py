# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
import json

# Set the source directory
base_dir = "./images/random10_result"
output_json = "image_labels.json"

# Store all records
data_list = []

for label_name in ["Complex", "NonComplex"]:
    label_value = 1 if label_name == "Complex" else 0
    folder_path = os.path.join(base_dir, label_name)
    if not os.path.isdir(folder_path):
        continue
    for subfolder in os.listdir(folder_path):
        final_path = os.path.join(folder_path, subfolder, "final_feature_map.png")
        if os.path.exists(final_path):
            data_list.append({"image_path": final_path, "label": label_value})

# Save as JSON
with open(output_json, "w") as f:
    json.dump(data_list, f, indent=4)

print(f"Saved {len(data_list)} entries to {output_json}")
