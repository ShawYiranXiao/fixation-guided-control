# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import requests
from PIL import Image
from io import BytesIO


def load_image(batch, input_key, output_key, params=None):
    # batch[output_key] = [Image.open(p).convert("RGB") for p in batch[input_key]]
    batch[output_key] = batch[input_key]
    return batch
