# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import AutoTokenizer
from PIL import Image
import argparse
import os


def load_pipeline(base_model_path, controlnet_checkpoint_path, torch_dtype=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch_dtype is None:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
    # Load ControlNet weights from checkpoint
    controlnet = ControlNetModel.from_pretrained(
        controlnet_checkpoint_path, torch_dtype=torch_dtype
    )

    # Load other components from the base model
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def load_conditioning_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))  # Required resolution
    return image


def run_inference(args):
    pipe = load_pipeline(args.base_model_path, args.controlnet_checkpoint)

    cond_image = load_conditioning_image(args.cond_image_path)

    print(f"Prompt: {args.prompt}")
    result = pipe(
        prompt=args.prompt,
        image=cond_image,
        num_inference_steps=args.steps,
        generator=torch.Generator(device="cpu").manual_seed(args.seed)
        if args.seed is not None
        else None,
    ).images[0]

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "result.png")
    result.save(output_path)
    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path", type=str, required=True, help="e.g., runwayml/stable-diffusion-v1-5"
    )
    parser.add_argument(
        "--controlnet_checkpoint",
        type=str,
        required=True,
        help="Path to your checkpoint-XXXXX/controlnet folder",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument(
        "--cond_image_path",
        type=str,
        required=True,
        help="Path to the control (conditioning) image",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Where to save the result"
    )
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")

    args = parser.parse_args()
    run_inference(args)
