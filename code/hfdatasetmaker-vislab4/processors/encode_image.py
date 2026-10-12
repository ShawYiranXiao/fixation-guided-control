# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
from datasets import Image as HFImage
from PIL import Image


def encode_image(batch, input_key, output_key, params=None):
    """Encode processed images as datasets.Image objects"""
    batch[output_key] = [HFImage().encode_example(img) for img in batch[input_key]]
    return batch
