# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
import glob
import argparse
from datasets import Dataset, Image


def combine_parquet(chunk_dir, output_dir, image_columns):
    """Merge Parquet chunks and save the dataset"""
    parquet_files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.parquet")))

    if not parquet_files:
        raise ValueError("No Parquet chunks found; run process_dataset.py first")

    dataset = Dataset.from_parquet(parquet_files)

    # **Cast the selected columns to datasets.Image**
    if image_columns:
        image_columns = image_columns.split(",")  # Comma-separated column names
        for column in image_columns:
            if column in dataset.column_names:
                dataset = dataset.cast_column(column, Image())
                print(f"Cast column `{column}` to datasets.Image")

    dataset.save_to_disk(output_dir)
    print(f"Merged {len(dataset)} records and saved them to {output_dir}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Combine Parquet chunks into a single dataset.")
    parser.add_argument(
        "--chunk-dir", type=str, required=True, help="Directory containing Parquet chunks"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save the combined dataset"
    )
    parser.add_argument(
        "--cast-image-columns",
        type=str,
        default="",
        help="Comma-separated list of columns to cast to Image()",
    )

    args = parser.parse_args()

    combine_parquet(args.chunk_dir, args.output_dir, args.cast_image_columns)


if __name__ == "__main__":
    main()
