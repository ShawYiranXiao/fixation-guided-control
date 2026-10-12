# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
from transformers import pipeline

caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")


def generate_caption(batch, input_key, output_key, params=None):
    batch[output_key] = [caption_pipeline(img)[0]["generated_text"] for img in batch[input_key]]
    return batch
