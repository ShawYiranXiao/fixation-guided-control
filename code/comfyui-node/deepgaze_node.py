# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
import torch
import requests
import base64
from PIL import Image
from io import BytesIO
import numpy as np


class DeepGazeRemoteNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Expected shape: [1, H, W, 3]
                "key": ("STRING", {"default": "sample_image"}),
                "num_points": ("INT", {"default": 4, "min": 1}),
                "total_iterations": ("INT", {"default": 4, "min": 1}),
                "centerbias": (["zeros", "original"],),
                "feature_method": (["mean", "softor"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "call_deepgaze"
    CATEGORY = "image"

    def call_deepgaze(self, image, key, num_points, total_iterations, centerbias, feature_method):
        # Convert tensor to base64 PNG
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)  # [H, W, 3]
        pil_img = Image.fromarray(img_np)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        payload = {
            "images": [img_b64],
            "keys": [key],
            "params": {
                "num_points": num_points,
                "total_iterations": total_iterations,
                "centerbias": centerbias,
                "feature_method": feature_method,
            },
        }

        try:
            response = requests.post(
                os.getenv("DEEPGAZE_SERVICE_URL", "http://127.0.0.1:5234/deepgaze"),
                json=payload,
                timeout=float(os.getenv("DEEPGAZE_REQUEST_TIMEOUT", "60")),
            )
            response.raise_for_status()
            result_b64 = response.json()["images"][0]
            result_img = Image.open(BytesIO(base64.b64decode(result_b64))).convert("RGB")

            # Convert back to tensor
            result_np = np.array(result_img).astype(np.float32) / 255.0
            result_tensor = torch.from_numpy(result_np).unsqueeze(0)  # [1, H, W, 3]

            return (result_tensor,)

        except Exception as e:
            print(f"DeepGaze service error: {e}")
            dummy = torch.zeros(1, 64, 64, 3)
            return (dummy,)


NODE_CLASS_MAPPINGS = {"DeepGazeRemoteNode": DeepGazeRemoteNode}

NODE_DISPLAY_NAME_MAPPINGS = {"DeepGazeRemoteNode": "DeepGaze via Flask"}
