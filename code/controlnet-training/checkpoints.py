# SPDX-License-Identifier: MIT
"""Resolve final models and nested optimizer-checkpoint model directories."""

from pathlib import Path


def resolve_controlnet_checkpoint(checkpoint):
    checkpoint = Path(checkpoint)
    model = checkpoint / "controlnet" if (checkpoint / "controlnet").is_dir() else checkpoint
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"ControlNet config.json not found in {model}")
    if not any(model.glob("*.safetensors")) and not any(model.glob("*.bin")):
        raise FileNotFoundError(f"ControlNet weights not found in {model}")
    return model
