# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter, maximum_filter

# === CLI arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--gray", action="store_true", help="Use grayscale colormap instead of jet")
args = parser.parse_args()


# === Output helpers ===
def save_colormap(heatmap, save_path, cmap="jet"):
    norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    colored = cm.get_cmap(cmap)(norm)[..., :3]
    img = Image.fromarray((colored * 255).astype(np.uint8))
    img.save(save_path)


def save_difference_map(method_name, fused, mean_map, base_name, diff_output_dir):
    diff = np.abs(fused - mean_map)
    diff_path = os.path.join(diff_output_dir, f"{base_name}_diff_vs_mean_{method_name}.jpg")
    save_colormap(diff, diff_path, cmap="hot")
    print(f"Saved difference map vs mean for {method_name} -> {diff_path}")


# === Fusion methods ===
def fusion_mean(heatmaps):
    return np.mean(heatmaps, axis=0)


def fusion_multiply_then_softmax(heatmaps):
    mul = np.prod(heatmaps, axis=0)
    exp = np.exp(mul - np.max(mul))
    return exp / np.sum(exp)


def fusion_softmax_then_multiply(heatmaps):
    softmaxed = np.exp(heatmaps - np.max(heatmaps, axis=(1, 2), keepdims=True))
    softmaxed /= np.sum(softmaxed, axis=(1, 2), keepdims=True)
    return np.prod(softmaxed, axis=0)


def fusion_softmax_then_multiply_then_softmax(heatmaps):
    softmaxed = np.exp(heatmaps - np.max(heatmaps, axis=(1, 2), keepdims=True))
    softmaxed /= np.sum(softmaxed, axis=(1, 2), keepdims=True)
    product = np.prod(softmaxed, axis=0)
    final_exp = np.exp(product - np.max(product))
    return final_exp / np.sum(final_exp)


def fusion_max(heatmaps):
    return np.max(heatmaps, axis=0)


def fusion_softor(heatmaps):
    softmaps = np.exp(heatmaps - np.max(heatmaps, axis=(1, 2), keepdims=True))
    softmaps /= np.sum(softmaps, axis=(1, 2), keepdims=True)
    complement = 1 - softmaps
    return 1 - np.prod(complement, axis=0)


def fusion_softmax_sum(heatmaps):
    softmaxed = np.exp(heatmaps - np.max(heatmaps, axis=(1, 2), keepdims=True))
    softmaxed /= np.sum(softmaxed, axis=(1, 2), keepdims=True)
    return np.sum(softmaxed, axis=0)


def fusion_softmax_lighten(heatmaps):
    softmaps = np.exp(heatmaps - np.max(heatmaps, axis=(1, 2), keepdims=True))
    softmaps /= np.sum(softmaps, axis=(1, 2), keepdims=True)
    return np.max(softmaps, axis=0)


def fusion_normalized_lighten(heatmaps):
    norm_maps = []
    for h in heatmaps:
        h_norm = (h - np.min(h)) / (np.max(h) - np.min(h) + 1e-8)
        norm_maps.append(h_norm)
    norm_maps = np.stack(norm_maps, axis=0)
    return np.max(norm_maps, axis=0)


def fusion_softmax_peak_preserve(heatmaps):
    """
    Preserve the peak regions of each normalized heatmap:
    - Normalize each map with softmax
    - Fuse the maps using a pixel-wise maximum (lighten)
    """
    softmaxed = np.exp(heatmaps - np.max(heatmaps, axis=(1, 2), keepdims=True))
    softmaxed /= np.sum(softmaxed, axis=(1, 2), keepdims=True)
    return np.max(softmaxed, axis=0)


def fusion_softmax_sum_enhanced(heatmaps):
    softmaxed = np.exp(heatmaps - np.max(heatmaps, axis=(1, 2), keepdims=True))
    softmaxed /= np.sum(softmaxed, axis=(1, 2), keepdims=True)
    summed = np.sum(softmaxed, axis=0)
    # Apply nonlinear enhancement to emphasize strong responses
    return np.power(summed, 1.5)


def fusion_weighted_softmax(heatmaps):
    weights = heatmaps.max(axis=(1, 2))
    weights /= weights.sum()
    return np.sum(heatmaps * weights[:, None, None], axis=0)


def fusion_gaussian_smooth(heatmaps):
    smoothed = np.array([gaussian_filter(h, sigma=3) for h in heatmaps])
    return np.mean(smoothed, axis=0)


def fusion_local_max(heatmaps):
    return np.max(np.array([h * (h == gaussian_filter(h, sigma=1)) for h in heatmaps]), axis=0)


def fusion_attention_like(heatmaps):
    weights = np.sum(heatmaps, axis=(1, 2))
    weights /= np.sum(weights)
    return np.sum(heatmaps * weights[:, None, None], axis=0)


def fusion_top_k_sum(heatmaps, percentile=99):
    masks = []
    for h in heatmaps:
        thresh = np.percentile(h, percentile)
        masks.append(np.where(h >= thresh, h, 0))
    fused = np.sum(masks, axis=0)
    return fused


def fusion_rank_weighted_sum(heatmaps):
    max_vals = np.max(heatmaps, axis=(1, 2))
    ranks = np.argsort(-max_vals)
    weights = np.linspace(1, 0.1, len(ranks))
    weights = weights[np.argsort(ranks)]
    return np.sum(heatmaps * weights[:, None, None], axis=0)


def fusion_adaptive_local_max(heatmaps, size=3):
    enhanced = []
    for h in heatmaps:
        peak_mask = h == maximum_filter(h, size=size)
        enhanced.append(h * peak_mask)
    return np.sum(enhanced, axis=0)


fusions = {
    "mean": fusion_mean,
    "fusion_max": fusion_max,
    "fusion_softmax_peak_preserve": fusion_softmax_peak_preserve,
    "fusion_softmax_sum_enhanced": fusion_softmax_sum_enhanced,
    "softor": fusion_softor,
    "fusion_softmax_sum": fusion_softmax_sum,
    "fusion_softmax_lighten": fusion_softmax_lighten,
    "fusion_normalized_lighten": fusion_normalized_lighten,
    "weighted_softmax": fusion_weighted_softmax,
    "fusion_top_k_sum": fusion_top_k_sum,
    "fusion_rank_weighted_sum": fusion_rank_weighted_sum,
    "fusion_adaptive_local_max": fusion_adaptive_local_max,
}

# === Image saving ===
npz_root_dir = os.getenv("DEEPGAZE_NPZ_ROOT", "./outputs/npz")
base_output_dir = os.getenv("DEEPGAZE_OUTPUT_ROOT", "./outputs")
output_root_dir = os.path.join(
    base_output_dir, "fusion_comparison_gray" if args.gray else "fusion_comparison_colormap"
)

# === {original, zeros} × {Complex, NonComplex} ===
for bias_type in ["original", "zeros"]:
    for group in ["Complex", "NonComplex"]:
        npz_dir = os.path.join(npz_root_dir, bias_type, group)
        output_dir = os.path.join(output_root_dir, bias_type, group)
        diff_output_dir = os.path.join(output_dir, "diffs")
        individual_softmax_dir = os.path.join(output_dir, "individual_softmax")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(diff_output_dir, exist_ok=True)
        os.makedirs(individual_softmax_dir, exist_ok=True)

        sample_files = sorted(
            f for f in os.listdir(npz_dir) if f.endswith(".npz") and not f.startswith("._")
        )

        for fname in sample_files:
            path = os.path.join(npz_dir, fname)
            try:
                data = np.load(path)["heatmaps"]
            except Exception as e:
                print(f"Failed to load {fname}: {e}")
                continue

            base = os.path.splitext(fname)[0]
            print(f"\n Processing [{bias_type}/{group}] {base}")

            # Individual softmax maps
            for i, h in enumerate(data):
                softmaxed = np.exp(h - np.max(h))
                softmaxed /= np.sum(softmaxed)
                individual_path = os.path.join(individual_softmax_dir, f"{base}_softmax_{i}.jpg")
                save_colormap(softmaxed, individual_path, cmap="gray" if args.gray else "jet")

            # Fusion visualizations
            mean_map = fusion_mean(data)
            n_methods = len(fusions)
            cols = 4
            rows = (n_methods + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = axes.flatten()

            for i, (method_name, fusion_func) in enumerate(fusions.items()):
                fused = fusion_func(data)
                out_path = os.path.join(output_dir, f"{base}_{method_name}.jpg")
                save_colormap(fused, out_path, cmap="gray" if args.gray else "jet")

                norm = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
                axes[i].imshow(norm, cmap="gray" if args.gray else "jet")
                axes[i].set_title(method_name)
                axes[i].axis("off")

                if method_name != "mean":
                    save_difference_map(method_name, fused, mean_map, base, diff_output_dir)

            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            grid_path = os.path.join(output_dir, f"{base}_grid.jpg")
            plt.savefig(grid_path, dpi=200)
            plt.close()
            print(f"Saved grid: {grid_path}")
