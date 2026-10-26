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
        commit_message="Release TypoFormer Stage 2 INT8 open-weights and standalone inference engine",
    )
    print(f"Successfully uploaded to Hugging Face Hub!")
    print(f"Repository URL: https://huggingface.co/{REPO_ID}")
    print(f"Commit: {commit_info}")


if __name__ == "__main__":
    upload()
