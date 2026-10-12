# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
from flask import Flask, request, jsonify
import base64
import numpy as np
import io
from PIL import Image
from processors.deepgaze_feature import deepgaze_process
from datasets import Image as HFImage

import time
import traceback

app = Flask(__name__)


@app.route("/deepgaze", methods=["POST"])
def run_deepgaze():
    try:
        # Parse request
        data = request.json
        images_b64 = data["images"]
        keys = data["keys"]
        params = data.get("params", {})

        # Decode images
        images = []
        for i, img_b64 in enumerate(images_b64):
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img).astype(np.float32)

            images.append(img_np)

        # Prepare batch
        batch = {
            "key": keys,
            "input": images,
        }

        # Run DeepGaze
        print(params)
        output_batch = deepgaze_process(
            batch, input_key="input", output_key="output", params=params
        )

        # Encode results
        result_b64 = []
        # for out_img in output_batch["output"]:
        #     out_img_uint8 = (np.clip(out_img, 0, 1) * 255).astype(np.uint8)
        #     pil_img = Image.fromarray(out_img_uint8)
        #     buf = io.BytesIO()
        #     pil_img.save(buf, format="PNG")
        #     result_b64.append(base64.b64encode(buf.getvalue()).decode())
        result_b64 = []

        for out_img in output_batch["output"]:
            if not isinstance(out_img, Image.Image):
                raise TypeError(f"Expected PIL.Image, got {type(out_img)}")

            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            result_b64.append(img_b64)

        return jsonify({"images": result_b64})

    except Exception as e:
        print("Exception in /deepgaze:", e)
        traceback.print_exc()  #  shows full stack trace in terminal
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5234)
