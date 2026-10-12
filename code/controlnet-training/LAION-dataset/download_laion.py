# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
from datasets import load_dataset
import pandas as pd

# Load the first 10,000 samples
dataset = load_dataset("laion/laion2B-en-aesthetic", split="train[:1500000]")


# Record filter
def is_valid(x):
    return (
        x.get("WIDTH") is not None
        and x.get("HEIGHT") is not None
        and x["WIDTH"] > 1024
        and x["HEIGHT"] > 768
        and x["WIDTH"] / x["HEIGHT"] > 1.32
        and x["WIDTH"] / x["HEIGHT"] < 1.34
        and x.get("URL") is not None
        and x.get("TEXT") is not None
    )


# Filter invalid records
filtered = dataset.filter(is_valid)

# Keep only the required fields
df = filtered.to_pandas()[["URL", "TEXT", "WIDTH", "HEIGHT"]]
df = df.rename(columns={"URL": "url", "TEXT": "text", "WIDTH": "width", "HEIGHT": "height"})


# Save as Parquet
df.to_parquet("laion_subset_filtered.parquet", index=False)
print(len(df))
print("Saved to laion_subset_filtered.parquet")
