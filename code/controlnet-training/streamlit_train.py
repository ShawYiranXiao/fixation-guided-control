# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import streamlit as st
import subprocess
import os
import time
from datetime import datetime

st.set_page_config(page_title="ControlNet Trainer", layout="wide")
st.title("ControlNet Streamlit Launcher")

# ===== 1. Training configuration =====
st.sidebar.header("Training parameters")
MODEL_DIR = st.sidebar.text_input("Pretrained Model Path", "runwayml/stable-diffusion-v1-5")
OUTPUT_DIR = st.sidebar.text_input("Output Directory", "./output/controlnet_run")
DATASET_NAME = st.sidebar.text_input("Dataset Name", "qiuweihao/mit_1003_zero_bias_landscapeonly")
RESUME = st.sidebar.text_input("Resume From Checkpoint (optional)", "")

RESOLUTION = st.sidebar.slider("Resolution", 128, 1024, 512, step=8)
LEARNING_RATE = st.sidebar.number_input("Learning Rate", value=4e-6, format="%.1e")
BATCH_SIZE = st.sidebar.number_input("Train Batch Size", min_value=1, value=8)
MAX_TRAIN_STEPS = st.sidebar.number_input("Max Train Steps", min_value=1000, value=20000, step=1000)
VALIDATION_STEPS = st.sidebar.number_input("Validation Steps", min_value=100, value=1000, step=100)

# ===== 2. Validation settings =====
st.sidebar.header("Validation settings")
PROMPTS = [
    "a group of people jumping on a beach",
    "a donkey pulling a cart",
    "a man is putting a bag of luggage",
]
IMAGES = [
    "./validation_images/NC-21-condition.png",
    "./validation_images/C-news-3-alter-condition.png",
    "./validation_images/C-evidence-3-condition.png",
]

# ===== 3. Training launch button =====
if st.button("Start training"):
    with st.expander("Training log", expanded=True):
        cmd = (
            [
                "accelerate",
                "launch",
                "train_controlnet.py",
                f"--pretrained_model_name_or_path={MODEL_DIR}",
                f"--output_dir={OUTPUT_DIR}",
                f"--dataset_name={DATASET_NAME}",
                "--conditioning_image_column=deepgaze_feature",
                "--image_column=image_original",
                "--caption_column=caption",
                f"--resolution={RESOLUTION}",
                f"--learning_rate={LEARNING_RATE}",
                "--lr_scheduler=cosine",
                "--lr_warmup_steps=500",
                f"--train_batch_size={BATCH_SIZE}",
                "--gradient_accumulation_steps=1",
                "--gradient_checkpointing",
                "--mixed_precision=fp16",
                "--validation_image",
            ]
            + IMAGES
            + ["--validation_prompt"]
            + PROMPTS
            + [
                f"--max_train_steps={MAX_TRAIN_STEPS}",
                f"--validation_steps={VALIDATION_STEPS}",
                f"--checkpointing_steps={VALIDATION_STEPS}",
                "--report_to=tensorboard",
            ]
        )

        if RESUME:
            cmd.append(f"--resume_from_checkpoint={RESUME}")

        st.code(" ".join(cmd), language="bash")
        st.success("Starting training...")

        # Start the training subprocess and display its output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        progress_placeholder = st.empty()
        step = 0

        for line in process.stdout:
            st.write(line.strip())
            if "step" in line and "/" in line:
                try:
                    step = int(line.split("step")[1].split("/")[0].strip())
                    progress_placeholder.progress(step / MAX_TRAIN_STEPS)
                except:
                    pass
            if "OSError" in line or "FileNotFoundError" in line:
                st.error(f"Unable to load the model or dataset: {line.strip()}")
                process.kill()
                break

        process.wait()
        if process.returncode == 0:
            st.success("Training completed!")
        else:
            st.error("Training failed or was interrupted.")
