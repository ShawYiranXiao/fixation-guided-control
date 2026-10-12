# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
from PIL import Image


def resize_keep_aspect_ratio(image, target_width, target_height):
    orig_width, orig_height = image.size
    scale = min(target_width / orig_width, target_height / orig_height)
    new_size = (int(orig_width * scale), int(orig_height * scale))
    resized_image = image.resize(new_size, Image.LANCZOS)
    return resized_image, new_size


def resize_max(image, target_width, target_height):
    resized, _ = resize_keep_aspect_ratio(image, target_width, target_height)
    return resized


def resize_padding(image, target_width, target_height, padding_color=(0, 0, 0)):
    resized, new_size = resize_keep_aspect_ratio(image, target_width, target_height)
    new_image = Image.new("RGB", (target_width, target_height), color=tuple(padding_color))
    offset_x = (target_width - new_size[0]) // 2
    offset_y = (target_height - new_size[1]) // 2
    new_image.paste(resized, (offset_x, offset_y))
    return new_image


def resize_stretch(image, target_width, target_height):
    return image.resize((target_width, target_height), Image.LANCZOS)


# def resize_max(image, max_width, max_height):
#     # Read original dimensions
#     orig_width, orig_height = image.size

#     # Compute the resize scale
#     scale_w = max_width / orig_width
#     scale_h = max_height / orig_height
#     scale = min(scale_w, scale_h)  # Keep the image within 1920 by 1080

#     # Resize to fit within 1920 by 1080
#     new_width = int(orig_width * scale)
#     new_height = int(orig_height * scale)

#     # Resize with high-quality interpolation
#     return image.resize((new_width, new_height), Image.LANCZOS)


def resize_image(batch, input_key, output_key, params):
    mode = params.get("mode", "max")
    target_width, target_height = params.get("target_resolution", [1920, 1080])
    padding_color = params.get("padding_color", [0, 0, 0])

    images = [img.convert("RGB") for img in batch[input_key]]

    if mode == "max":
        batch[output_key] = [resize_max(img, target_width, target_height) for img in images]
    elif mode == "stretch":
        batch[output_key] = [resize_stretch(img, target_width, target_height) for img in images]
    elif mode == "padding":
        batch[output_key] = [
            resize_padding(img, target_width, target_height, padding_color) for img in images
        ]
    else:
        raise ValueError(f"Unsupported resize mode: {mode}")

    return batch
