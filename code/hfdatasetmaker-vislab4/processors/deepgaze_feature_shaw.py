# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
import torch
from PIL import Image as PILImage
from datasets import Image as HFImage
import deepgaze_pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage import zoom
from scipy.special import logsumexp
import matplotlib.pyplot as plt

centerbias_template = np.load(os.getenv("DEEPGAZE_CENTERBIAS_PATH", "centerbias_mit1003.npy"))


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
    normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min())

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


def process_image_with_deepgaze_batch(image, num_points, batch_size, total_iterations):
    image_np = np.array(image)  #  Convert the image to a NumPy array
    print(image_np.shape)
    H, W = image_np.shape[:2]  # Read the image height and width
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
        print(
            f"Iteration: {iteration_counter} to {iteration_counter + cur_batch_size} | Total Iterations: {total_iterations}"
        )

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
        print("enter model")
        print(image_tensor.shape)
        log_density_predictions = model(
            image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor
        )  # Shape: (batch, 1, H, W)
        print("exit model")
        # Extract heatmaps and compute their mean
        heatmaps = log_density_predictions.detach().cpu().numpy()[:, 0]  # Shape: (batch, H, W)
        print(heatmaps.shape)

        batch_heatmaps.append(heatmaps)
        iteration_counter += cur_batch_size

    batch_heatmaps = np.concatenate(batch_heatmaps, axis=0)  # (total_iterations, H, W)
    print(batch_heatmaps.shape)

    final_heatmap = np.mean(batch_heatmaps, axis=0)  # Shape: (H, W)
    print(final_heatmap.shape)

    # Normalize the heatmap to the range [0, 255]
    # final_heatmap = (255 * (final_heatmap - np.min(final_heatmap)) / (np.max(final_heatmap) - np.min(final_heatmap))).astype(np.uint8)
    return final_heatmap


def deepgaze_process(
    batch, input_key, output_key, num_points=4, batch_random_size=1, total_iterations=10
):
    result = []
    for image in batch[input_key]:
        print("enter batch")
        heatmap = process_image_with_deepgaze(image, num_points, total_iterations)
        print(type(heatmap))

        heatmap_image = apply_colormap(heatmap)
        print(type(heatmap_image))
        result.append(heatmap_image)
    batch[output_key] = result
    return batch


# # Shaw's original code
def process_image_with_deepgaze(image, num_points, num_iterations):
    image_np = np.array(image)
    image_height, image_width = image_np.shape[:2]
    # rescale to match image size
    centerbias = zoom(
        centerbias_template,
        (image_height / centerbias_template.shape[0], image_width / centerbias_template.shape[1]),
        order=0,
        mode="nearest",
    )
    # renormalize log density
    centerbias -= logsumexp(centerbias)

    centerbias = np.zeros([image_np.shape[0], image_np.shape[1]])
    centerbias_tensor = torch.tensor([centerbias], device=device)  # (batch, H, W)

    heatmaps = []

    for _ in range(num_iterations):
        # Generate random fixation points that cover the entire image
        points = [
            (np.random.randint(0, image_width), np.random.randint(0, image_height))
            for _ in range(num_points)
        ]

        # Extract x and y coordinates of the fixation points
        fixation_history_x = np.array([p[0] for p in points])
        fixation_history_y = np.array([p[1] for p in points])

        # Convert the inputs to tensors
        image_tensor = torch.tensor([image_np.transpose(2, 0, 1)]).float().to(device)
        centerbias_tensor = torch.tensor([centerbias]).float().to(device)
        x_hist_tensor = torch.tensor([fixation_history_x]).float().to(device)
        y_hist_tensor = torch.tensor([fixation_history_y]).float().to(device)

        # Generate the log density prediction
        log_density_prediction = model(
            image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor
        )
        predicted_heatmap = log_density_prediction.detach().cpu().numpy()[0, 0]
        heatmaps.append(predicted_heatmap)

    # Average the heatmaps to reduce the influence of individual fixation points
    final_heatmap = np.mean(heatmaps, axis=0)
    return final_heatmap
