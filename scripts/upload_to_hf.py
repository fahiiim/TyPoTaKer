"""Upload standalone TypoFormer open-weight model repository to Hugging Face Hub."""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, get_token

# Retrieve token from environment, huggingface cache, or explicit file
TOKEN = os.environ.get("HF_TOKEN") or get_token()
if not TOKEN:
    cache_token = Path.home() / ".cache" / "huggingface" / "token"
    if cache_token.exists():
        TOKEN = cache_token.read_text(encoding="utf-8").strip()

if not TOKEN:
    print("ERROR: Hugging Face token not found! Please set HF_TOKEN or run huggingface-cli login.")
    sys.exit(1)

REPO_ID = "fahiiim/TypoFormer"
LOCAL_DIR = Path("hf_dist").resolve()


def upload():
    print(f"Connecting to Hugging Face Hub for repo: {REPO_ID}...")
    api = HfApi(token=TOKEN)
    user_info = api.whoami()
    print(f"Authenticated as: {user_info.get('name')}")
    
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    # 1. Clean up obsolete pycache if present on remote
    try:
        remote_files = [s.rfilename for s in api.repo_info(REPO_ID, repo_type="model").siblings]
        for f in remote_files:
            if "__pycache__" in f:
                print(f"Deleting obsolete remote artifact: {f}...")
                api.delete_file(path_in_repo=f, repo_id=REPO_ID, repo_type="model", commit_message=f"Clean up {f}")
    except Exception as e:
        print(f"Note on cleanup check: {e}")

    # Ensure local __pycache__ does not exist before upload
    pycache = LOCAL_DIR / "__pycache__"
    if pycache.exists():
        import shutil
        shutil.rmtree(pycache, ignore_errors=True)

    # 2. Upload distribution folder
    print(f"Uploading distribution folder {LOCAL_DIR} to {REPO_ID}...")
    commit_info = api.upload_folder(
        folder_path=str(LOCAL_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Release TypoFormer Stage 2 INT8 with Model Architecture Diagram & Polished Model Card",
    )
    print("Distribution folder successfully uploaded to Hugging Face Hub!")

    # 3. Check if release zip needs upload
    zip_file = Path("installer/TypoFormer-win-x64.zip").resolve()
    if zip_file.exists():
        print(f"Verifying standalone release zip {zip_file} on {REPO_ID}...")
        api.upload_file(
            path_or_fileobj=str(zip_file),
            path_in_repo="TypoFormer-win-x64.zip",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Update TypoFormer-win-x64.zip release artifact",
        )
        print("Release ZIP verified/uploaded successfully!")

    print(f"\nSuccessfully published to Hugging Face Hub!")
    print(f"Repository URL: https://huggingface.co/{REPO_ID}")
    print(f"Commit: {commit_info}")

    # Verify remote file listing
    print("\nRemote repository files:")
    info = api.repo_info(REPO_ID, repo_type="model")
    for s in info.siblings:
        print(f"  - {s.rfilename}")


if __name__ == "__main__":
    upload()
