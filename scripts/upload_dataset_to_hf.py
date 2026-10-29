"""Upload TypoFormer synthetic dataset to Hugging Face Hub as a dedicated dataset."""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, get_token

TOKEN = os.environ.get("HF_TOKEN") or get_token()
if not TOKEN:
    cache_token = Path.home() / ".cache" / "huggingface" / "token"
    if cache_token.exists():
        TOKEN = cache_token.read_text(encoding="utf-8").strip()

if not TOKEN:
    print("ERROR: Hugging Face token not found!")
    sys.exit(1)

REPO_ID = "fahiiim/TypoFormer-Synthetic-Dataset"
LOCAL_DIR = Path("dataset_dist").resolve()


def upload():
    print(f"Connecting to Hugging Face Hub for dataset: {REPO_ID}...")
    api = HfApi(token=TOKEN)
    user_info = api.whoami()
    print(f"Authenticated as: {user_info.get('name')}")

    print(f"Creating dataset repo {REPO_ID} (if not already existing)...")
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    print(f"Uploading dataset folder {LOCAL_DIR} to {REPO_ID}...")
    commit_info = api.upload_folder(
        folder_path=str(LOCAL_DIR),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Publish TypoFormer 53k synthetic dataset for GEC, Typo Repair, and Autocomplete",
    )

    print("\nDataset successfully published to Hugging Face Hub!")
    print(f"Dataset URL: https://huggingface.co/datasets/{REPO_ID}")
    print(f"Commit: {commit_info}")

    print("\nRemote dataset files:")
    info = api.dataset_info(REPO_ID)
    for s in info.siblings:
        print(f"  - {s.rfilename}")


if __name__ == "__main__":
    upload()
