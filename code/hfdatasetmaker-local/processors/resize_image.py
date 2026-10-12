# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
from PIL import Image


def resize_max(image, max_width, max_height):
    # Read original dimensions
    orig_width, orig_height = image.size

    # Compute the resize scale
    scale_w = max_width / orig_width
    scale_h = max_height / orig_height
    scale = min(scale_w, scale_h)  # Keep the image within 1920 by 1080

    # Resize to fit within 1920 by 1080
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    # Resize with high-quality interpolation
    return image.resize((new_width, new_height), Image.LANCZOS)


def resize_image(batch, input_key, output_key, params={"max_dims": [1920, 1080]}):
    max_dims = params["max_dims"]
    batch[output_key] = [
        resize_max(img, max_width=max_dims[0], max_height=max_dims[1]) for img in batch[input_key]
    ]
    return batch
