"""Upload standalone TypoFormer open-weight model repository to Hugging Face Hub."""

import os
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "fahiiim/TypoFormer"
LOCAL_DIR = Path("hf_dist").resolve()


def upload():
    print(f"Connecting to Hugging Face Hub with token for {REPO_ID}...")
    api = HfApi(token=TOKEN)
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    print(f"Uploading distribution folder {LOCAL_DIR} to {REPO_ID}...")
    commit_info = api.upload_folder(
        folder_path=str(LOCAL_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Release TypoFormer Stage 2 INT8 Wine-Tuned open-weights and standalone inference engine",
    )
    print(f"Distribution folder uploaded to Hugging Face Hub!")

    zip_file = Path("installer/TypoFormer-win-x64.zip").resolve()
    if zip_file.exists():
        print(f"Uploading standalone release zip {zip_file} to {REPO_ID}...")
        api.upload_file(
            path_or_fileobj=str(zip_file),
            path_in_repo="TypoFormer-win-x64.zip",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Update TypoFormer-win-x64.zip with fine-tuned Wine models",
        )
        print(f"Release ZIP uploaded successfully!")

    print(f"Repository URL: https://huggingface.co/{REPO_ID}")
    print(f"Commit: {commit_info}")


if __name__ == "__main__":
    upload()
