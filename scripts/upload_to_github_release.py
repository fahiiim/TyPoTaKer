"""Upload latest TypoFormer-win-x64.zip to GitHub Releases."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
import urllib.error
import urllib.request


def get_github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    # Try git credential helper
    p = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n",
        text=True,
        capture_output=True,
    )
    for line in p.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("No GitHub token found in environment or git credentials")


def upload_release_asset():
    token = get_github_token()
    repo = "fahiiim/TyPoTaKer"
    zip_path = Path("installer/TypoFormer-win-x64.zip").resolve()
    
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
        
    size = zip_path.stat().st_size
    print(f"Preparing to upload {zip_path.name} ({size / (1024*1024):.2f} MB) to {repo} release...")

    headers = {
        "Authorization": f"token {token}",
        "User-Agent": "TypoFormer-Uploader",
        "Accept": "application/vnd.github.v3+json",
    }

    # 1. Get latest release
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases", headers=headers)
    with urllib.request.urlopen(req) as resp:
        releases = json.load(resp)
        
    if not releases:
        raise RuntimeError("No releases found")
        
    release = releases[0]
    release_id = release["id"]
    tag = release["tag_name"]
    print(f"Target release: {release['name']} (ID: {release_id}, Tag: {tag})")

    # 2. Check existing assets and delete old zip if present
    for asset in release.get("assets", []):
        if asset["name"] == zip_path.name:
            asset_id = asset["id"]
            print(f"Deleting existing outdated asset {asset['name']} (ID: {asset_id}, Size: {asset['size']})...")
            del_req = urllib.request.Request(
                f"https://api.github.com/repos/{repo}/releases/assets/{asset_id}",
                headers=headers,
                method="DELETE",
            )
            with urllib.request.urlopen(del_req) as del_resp:
                print(f"Deleted old asset. Status: {del_resp.status}")
            time.sleep(1)
            break

    # 3. Upload new asset
    upload_url = release["upload_url"].split("{")[0] + f"?name={zip_path.name}"
    print(f"Uploading new asset to {upload_url}...")

    upload_headers = {
        "Authorization": f"token {token}",
        "User-Agent": "TypoFormer-Uploader",
        "Content-Type": "application/zip",
        "Content-Length": str(size),
    }

    with open(zip_path, "rb") as f:
        upload_req = urllib.request.Request(
            upload_url,
            data=f,
            headers=upload_headers,
            method="POST",
        )
        with urllib.request.urlopen(upload_req) as up_resp:
            result = json.load(up_resp)
            print(f"[OK] Successfully uploaded new asset!")
            print(f"  Asset ID: {result['id']}")
            print(f"  Size:     {result['size']} bytes ({result['size'] / (1024*1024):.2f} MB)")
            print(f"  Download: {result['browser_download_url']}")


if __name__ == "__main__":
    upload_release_asset()
