# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import argparse
from huggingface_hub import HfApi
import os
from datasets import load_from_disk


def load_local_dataset(dataset_path: str):
    """
    Load the local dataset
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset `{dataset_path}` does not exist; check the path!")

    print("Load the local dataset...")
    dataset = load_from_disk(dataset_path)

    print(f"Dataset loaded successfully!")
    print(dataset)
    print(dataset[0])  # Print the first record

    return dataset


def upload_to_huggingface(dataset, hf_username: str, hf_repo_name: str, hf_token: str):
    """
    Upload the dataset to Hugging Face Hub
    """
    if not hf_username or not hf_repo_name:
        raise ValueError("Set HF_USERNAME and a dataset repository name before uploading.")
    if hf_token is None:
        raise ValueError("Set the HF_TOKEN environment variable to upload data")

    api = HfApi(token=hf_token)
    hf_dataset_repo = f"{hf_username}/{hf_repo_name}"

    #  Check whether the Hugging Face repository exists
    try:
        existing_datasets = api.list_datasets(author=hf_username)
        existing_repo_ids = set(repo.id for repo in existing_datasets if hasattr(repo, "id"))
    except Exception as e:
        print(f"Unable to list Hugging Face datasets: {e}")
        existing_repo_ids = set()  # Assume no datasets are available

    #  Create the dataset repository if it does not exist
    if hf_dataset_repo not in existing_repo_ids:
        print(f"Creating dataset repository `{hf_dataset_repo}`...")
        api.create_repo(repo_id=hf_dataset_repo, repo_type="dataset", exist_ok=True)
        print(f"Created dataset repository `{hf_dataset_repo}`!")
    else:
        print(f"Dataset repository `{hf_dataset_repo}` already exists; continuing upload...")

    #  Upload the dataset to Hugging Face Hub
    print("Uploading data to Hugging Face Hub...")
    dataset.push_to_hub(hf_dataset_repo, token=hf_token)
    print(f"Dataset uploaded to: https://huggingface.co/datasets/{hf_dataset_repo}")


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub.")
    parser.add_argument(
        "--dataset-dir", type=str, required=True, help="Path to the dataset directory"
    )
    parser.add_argument("--repo-name", type=str, required=True, help="Hugging Face repository name")
    parser.add_argument(
        "--hf-token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face API Token"
    )
    parser.add_argument(
        "--hf-username", type=str, default=os.getenv("HF_USERNAME"), help="Hugging Face username"
    )

    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError("Provide --hf-token or set the HF_TOKEN environment variable")

    dataset = load_local_dataset(args.dataset_dir)
    upload_to_huggingface(dataset, args.hf_username, args.repo_name, args.hf_token)


if __name__ == "__main__":
    main()
