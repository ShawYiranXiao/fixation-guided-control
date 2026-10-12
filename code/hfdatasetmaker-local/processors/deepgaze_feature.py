# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import torch
from PIL import Image as PILImage
from datasets import Image as HFImage
import deepgaze_pytorch
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage import zoom
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import os
import sys
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deepgaze_model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)

centerbias_template_original = np.load(
    os.getenv("DEEPGAZE_CENTERBIAS_PATH", "centerbias_mit1003.npy")
)
centerbias_template_zeros = np.zeros(
    [centerbias_template_original.shape[0], centerbias_template_original.shape[1]]
)


def fusion_mean(heatmaps):
    return np.mean(heatmaps, axis=0)


def fusion_softor(heatmaps):
    softmaps = np.exp(heatmaps - np.max(heatmaps, axis=(1, 2), keepdims=True))
    softmaps /= np.sum(softmaps, axis=(1, 2), keepdims=True)
    complement = 1 - softmaps
    return 1 - np.prod(complement, axis=0)


def fusion_weighted_softmax(heatmaps):
    weights = heatmaps.max(axis=(1, 2))
    weights /= weights.sum()
    return np.sum(heatmaps * weights[:, None, None], axis=0)


fusion_methods = {
    "mean": fusion_mean,
    "softor": fusion_softor,
    "weighted_softmax": fusion_weighted_softmax,
}


def apply_colormap(image_array, colormap="jet"):
    """
    Converts a grayscale NumPy array into a colored image using a colormap.

    Args:
        image_array (np.ndarray): Input grayscale image (H, W) or (H, W, 1)
        colormap (str): Colormap name (e.g., "jet", "viridis", "magma")

    Returns:
        PIL.Image: Colormapped image
    """
    if len(image_array.shape) == 3 and image_array.shape[2] == 1:
        image_array = image_array.squeeze()  # Convert (H, W, 1) -> (H, W)

    # Normalize to [0,1] if not already
    normalized = (image_array - image_array.min()) / (
        image_array.max() - image_array.min() + 1e-8
    )  # add eps for stability

    # Apply the colormap
    colormap_function = cm.get_cmap(colormap)
    colored_image = colormap_function(normalized)  # (H, W, 4) -> RGBA

    # Convert to uint8 and remove the alpha channel
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)  # RGB format

    # Convert to PIL image
    return PILImage.fromarray(colored_image)


def normalize_image_array_8bit(image):
    normalized_image = (
        255 * (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    ).astype(np.uint8)
    return normalized_image


# helper to save npz
def save_heatmaps_npz(batch_heatmaps, npz_output_dir, dataset_key):
    os.makedirs(npz_output_dir, exist_ok=True)

    if not isinstance(dataset_key, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", dataset_key
    ):
        raise ValueError("Dataset keys must be safe filename identifiers.")
    npz_path = os.path.join(npz_output_dir, f"{dataset_key}_heatmaps.npz")

    np.savez_compressed(npz_path, heatmaps=batch_heatmaps)
    print(f"Saved raw heatmaps to -> {npz_path}")


def process_image_with_deepgaze_batch(
    *,
    model,
    centerbias_template,
    image,
    num_points,
    batch_size,
    total_iterations,
    dataset_key=None,
    npz_output_dir=None,
    feature_method="mean",
):

    image_np = np.array(image)  #  Convert the image to a NumPy array
    # print(image_np.shape)
    H, W = image_np.shape[:2]  # Read the image height and width

    if not dataset_key:
        raise ValueError("Dataset key is missing. Unable to process image without key.")

    if not npz_output_dir:
        raise ValueError("Output directory for npz files is not provided.")

    if not isinstance(dataset_key, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", dataset_key
    ):
        raise ValueError("Dataset keys must be safe filename identifiers.")
    npz_path = os.path.join(npz_output_dir, f"{dataset_key}_heatmaps.npz")
    if os.path.exists(npz_path):
        print(f"Found existing .npz → Loading: {npz_path}")
        heatmaps = np.load(npz_path)["heatmaps"]
    else:
        print(f"Generating heatmaps for {dataset_key}...")
        # rescale to match image size
        centerbias = zoom(
            centerbias_template,
            (H / centerbias_template.shape[0], W / centerbias_template.shape[1]),
            order=0,
            mode="nearest",
        )
        # renormalize log density
        centerbias -= logsumexp(centerbias)
        centerbias_tensor = torch.tensor([centerbias], device=device)  # (batch, H, W)

        iteration_counter = 0
        batch_heatmaps = []

        while iteration_counter < total_iterations:
            cur_batch_size = min(batch_size, total_iterations - iteration_counter)
            # print(f"Iteration: {iteration_counter} to {iteration_counter + cur_batch_size} | Total Iterations: {total_iterations}")

            # Sample random fixation histories for each iteration
            points = np.random.randint(0, [W, H], size=(cur_batch_size, num_points, 2))
            fixation_history_x = points[:, :, 0]  # Shape: (num_iterations, num_points)
            fixation_history_y = points[:, :, 1]

            # Batch the center bias and image tensors
            image_batch = np.repeat(
                image_np[None, :, :, :], cur_batch_size, axis=0
            )  # Shape: (batch_size, H, W, C)
            # Convert to PyTorch tensors on the selected device
            image_tensor = torch.tensor(
                image_batch.transpose(0, 3, 1, 2), device=device
            )  # (batch, C, H, W)
            x_hist_tensor = torch.tensor(fixation_history_x, device=device)  # (batch, num_points)
            y_hist_tensor = torch.tensor(fixation_history_y, device=device)  # (batch, num_points)

            # Run batched predictions
            log_density_predictions = model(
                image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor
            )  # Shape: (batch, 1, H, W)
            # Extract heatmaps and compute their mean
            heatmaps = log_density_predictions.detach().cpu().numpy()[:, 0]  # Shape: (batch, H, W)
            batch_heatmaps.append(heatmaps)
            iteration_counter += cur_batch_size

        heatmaps = np.concatenate(batch_heatmaps, axis=0)  # (total_iterations, H, W)

        save_heatmaps_npz(heatmaps, npz_output_dir, dataset_key)

    # fusion method support
    if feature_method in fusion_methods:
        final_heatmap = fusion_methods[feature_method](heatmaps)
    else:
        raise ValueError(f"Unsupported feature_method: {feature_method}")

    # Normalize the heatmap to the range [0, 255]
    final_heatmap = (
        255
        * (final_heatmap - np.min(final_heatmap))
        / (np.max(final_heatmap) - np.min(final_heatmap))
    ).astype(np.uint8)
    return final_heatmap


def deepgaze_process(batch, input_key, output_key, params):
    num_points = params.get("num_points", 4)
    batch_random_size = params.get("batch_random_size", 1)
    total_iterations = params.get("total_iterations", 4)
    centerbias_type = params.get("centerbias", "zeros")
    feature_method = params.get("feature_method", "mean")
    npz_output_dir = params.get("npz_output_dir", "diff_output/npz_heatmaps")
    os.makedirs(npz_output_dir, exist_ok=True)

    result = []

    for i, image in enumerate(batch[input_key]):
        centerbias_template = (
            centerbias_template_zeros
            if centerbias_type == "zeros"
            else centerbias_template_original
        )

        # dataset_key = batch["key"][i]
        # dataset_key = batch["key"][i] if "key" in batch else f"{i}"
        if "key" not in batch:
            raise ValueError(
                "Missing 'key' field in batch. Please ensure the dataset includes a 'key' column."
            )

        dataset_key = batch["key"][i]

        params_for_batch = {
            "model": deepgaze_model,
            "centerbias_template": centerbias_template,
            "image": image,
            "num_points": num_points,
            "batch_size": batch_random_size,
            "total_iterations": total_iterations,
            "dataset_key": dataset_key,
            "npz_output_dir": npz_output_dir,
            "feature_method": feature_method,
        }

        heatmap = process_image_with_deepgaze_batch(**params_for_batch)
        heatmap_image = apply_colormap(heatmap)
        result.append(heatmap_image)

    batch[output_key] = result
    return batch


# # Shaw's original code
# def process_image_with_deepgaze(image, num_points, num_iterations):
#     image_np = np.array(image)
#     image_height, image_width = image_np.shape[:2]
#     # rescale to match image size
#     centerbias = zoom(centerbias_template, (image_height/centerbias_template.shape[0], image_width/centerbias_template.shape[1]), order=0, mode='nearest')
#     # renormalize log density
#     centerbias -= logsumexp(centerbias)

#     centerbias = np.zeros([image_np.shape[0], image_np.shape[1]])
#     centerbias_tensor = torch.tensor([centerbias], device=device)  # (batch, H, W)

#     heatmaps= []

#     for _ in range(num_iterations):
#         # Generate random fixation points that cover the entire image
#         points = [(np.random.randint(0, image_width), np.random.randint(0, image_height)) for _ in range(num_points)]

#         # Extract x and y coordinates of the fixation points
#         fixation_history_x = np.array([p[0] for p in points])
#         fixation_history_y = np.array([p[1] for p in points])

#         # Convert the inputs to tensors
#         image_tensor = torch.tensor([image_np.transpose(2, 0, 1)]).float().to(device)
#         centerbias_tensor = torch.tensor([centerbias]).float().to(device)
#         x_hist_tensor = torch.tensor([fixation_history_x]).float().to(device)
#         y_hist_tensor = torch.tensor([fixation_history_y]).float().to(device)

#         # Generate the log density prediction
#         log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
#         predicted_heatmap = log_density_prediction.detach().cpu().numpy()[0, 0]
#         heatmaps.append(predicted_heatmap)

#     # Average the heatmaps to reduce the influence of individual fixation points
#     final_heatmap = np.mean(heatmaps, axis=0)
#     return final_heatmap
