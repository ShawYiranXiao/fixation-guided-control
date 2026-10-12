# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
# streamlit_inference_model.py
import streamlit as st
from PIL import Image
import torch
import os
import sys
import json
from pathlib import Path
from checkpoints import resolve_controlnet_checkpoint
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# --- CLI args ---
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint_root",
    type=str,
    default="./output_dir",
    help="Root directory containing checkpoints",
)
args, _ = parser.parse_known_args()

checkpoint_root = Path(args.checkpoint_root)

st.title("ControlNet Checkpoint Inference")
st.markdown(f"Checkpoint root: `{checkpoint_root}`")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED_BASE = "runwayml/stable-diffusion-v1-5"

# --- Find checkpoints ---
if not checkpoint_root.exists():
    st.error(f"Checkpoint root `{checkpoint_root}` does not exist.")
    st.stop()

checkpoints = [f.name for f in checkpoint_root.iterdir() if f.is_dir()]
if not checkpoints:
    st.error("No checkpoints found.")
    st.stop()

checkpoint = st.selectbox("Select a Checkpoint", sorted(checkpoints))
checkpoint_path = checkpoint_root / checkpoint

# --- Upload image ---
uploaded_file = st.file_uploader("Upload validation image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    validation_image = Image.open(uploaded_file).convert("RGB")
    st.image(validation_image, caption="Validation Image", use_column_width=True)
else:
    st.warning("Please upload a conditioning image.")

# --- Prompt input ---
prompt = st.text_input("Enter your prompt", value="a donkey pulling a cart")

# --- Run inference ---
if st.button("Run Inference") and uploaded_file and prompt:
    st.info("Running inference...")

    try:
        model_path = resolve_controlnet_checkpoint(checkpoint_path)
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    weight_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    controlnet = ControlNetModel.from_pretrained(model_path, torch_dtype=weight_dtype)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        PRETRAINED_BASE,
        controlnet=controlnet,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=DEVICE).manual_seed(42)
    with torch.autocast("cuda") if DEVICE == "cuda" else torch.no_grad():
        output = pipe(
            prompt, image=validation_image, num_inference_steps=20, generator=generator
        ).images[0]

    st.image(output, caption="Generated Image", use_column_width=True)
