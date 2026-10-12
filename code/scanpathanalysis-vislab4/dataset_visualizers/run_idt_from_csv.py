# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import csv
import json
import argparse
import numpy as np
import os


def apply_idt(x, y, timestamps, dispersion_threshold=25, min_fixation_duration=0.1):
    """
    Streamlit-equivalent I-DT fixation detection.

    Args:
        x, y: gaze coordinates (lists)
        timestamps: list of timestamps in seconds
        dispersion_threshold: max x/y range in pixels for a fixation
        min_fixation_duration: minimum duration (in seconds)

    Returns:
        List of fixation dicts with x, y, start_time, end_time, duration
    """
    fixations = []
    i = 0
    n = len(x)

    while i < n:
        j = i
        while (
            j < n
            and max(x[i : j + 1]) - min(x[i : j + 1]) < dispersion_threshold
            and max(y[i : j + 1]) - min(y[i : j + 1]) < dispersion_threshold
        ):
            j += 1

        duration = timestamps[j - 1] - timestamps[i] if j > i else 0

        if duration >= min_fixation_duration:
            fx = float(np.mean(x[i:j]))
            fy = float(np.mean(y[i:j]))
            fixations.append(
                {
                    "x": fx,
                    "y": fy,
                    "start_time": float(timestamps[i]),
                    "end_time": float(timestamps[j - 1]),
                    "duration": float(duration),
                }
            )

        i = j

    return fixations


def load_csv(path):
    timestamps, x, y = [], [], []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            t, xi, yi = map(float, row[:3])
            timestamps.append(t)
            x.append(xi)
            y.append(yi)
    return timestamps, x, y


def save_json(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Fixations saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to input CSV (timestamp,x,y)")
    parser.add_argument("--out", default="fixations.json", help="Output JSON file")
    parser.add_argument(
        "--dispersion", type=float, default=0.05, help="Dispersion threshold in pixels"
    )
    parser.add_argument(
        "--duration", type=float, default=0.1, help="Min fixation duration in seconds"
    )
    args = parser.parse_args()

    timestamps, x, y = load_csv(args.csv)
    fixations = apply_idt(x, y, timestamps, args.dispersion, args.duration)
    save_json(fixations, args.out)
