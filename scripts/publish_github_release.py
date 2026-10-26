"""
Automatically creates the latest GitHub Release and uploads TypoFormer-win-x64.zip
using credentials from git credential manager.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path


def get_github_token() -> str:
    proc = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n",
        capture_output=True,
        text=True,
        check=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("Could not find GitHub token from git credential manager")


def create_release_and_upload():
    root = Path(__file__).resolve().parent.parent
    zip_path = root / "installer" / "TypoFormer-win-x64.zip"

    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}. Run scripts/package_release.py first.")

    token = get_github_token()
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "TypoFormer-Release-Bot",
    }

    tag_name = "v1.0.0"
    repo = "fahiiim/TyPoTaKer"

    print(f"Checking if release {tag_name} exists on {repo}...")
    release_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag_name}"
    req = urllib.request.Request(release_url, headers=headers)
    release_data = None
    try:
        with urllib.request.urlopen(req) as resp:
            release_data = json.loads(resp.read().decode())
            print(f"Found existing release: ID {release_data['id']}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("Release does not exist. Creating new release...")
        else:
            raise

    if not release_data:
        create_url = f"https://api.github.com/repos/{repo}/releases"
        payload = {
            "tag_name": tag_name,
            "target_commitish": "main",
            "name": "TypoFormer v1.0.0 — Production Release (Offline AI Keystroke Companion)",
            "body": (
                "## TypoFormer v1.0.0 (Production Release)\n\n"
                "**TypoFormer** is a high-precision, character-aware hybrid transformer for real-time edge autocomplete "
                "and non-autoregressive grammar error correction (GEC).\n\n"
                "### Included in this release:\n"
                "- Standalone .NET 8 WPF desktop application with embedded fountain pen branding\n"
                "- Stage 2 INT8 Causal Language Model (lm_stage2_int8.onnx — 94M parameters)\n"
                "- Stage 2 INT8 GEC Edit-Tagger (gec_stage2_int8.onnx)\n"
                "- Complete 16,000 Byte-Level BPE tokenizers and edit-confusion vocabulary maps\n"
                "- QuillBot-style Grammar Assistant with real-time error cards, gutter counts, and floating fix overlay\n\n"
                "### 1-Line PowerShell Install:\n"
                "```powershell\n"
                "irm https://raw.githubusercontent.com/fahiiim/TyPoTaKer/main/install.ps1 | iex\n"
                "```\n"
            ),
            "draft": False,
            "prerelease": False,
        }
        create_req = urllib.request.Request(
            create_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
        )
        with urllib.request.urlopen(create_req) as resp:
            release_data = json.loads(resp.read().decode())
            print(f"[OK] Created release: ID {release_data['id']} at {release_data['html_url']}")

    release_id = release_data["id"]
    upload_url_template = release_data["upload_url"]  # format: https://uploads.github.com/.../assets{?name,label}
    upload_base = upload_url_template.split("{")[0]

    # Check if asset already exists
    assets_url = f"https://api.github.com/repos/{repo}/releases/{release_id}/assets"
    req_assets = urllib.request.Request(assets_url, headers=headers)
    with urllib.request.urlopen(req_assets) as resp:
        existing_assets = json.loads(resp.read().decode())

    asset_name = zip_path.name
    for a in existing_assets:
        if a["name"] == asset_name:
            print(f"Deleting older asset ID {a['id']} ({a['name']})...")
            del_req = urllib.request.Request(
                f"https://api.github.com/repos/{repo}/releases/assets/{a['id']}",
                headers=headers,
                method="DELETE",
            )
            with urllib.request.urlopen(del_req) as del_resp:
                pass
            print("[OK] Deleted previous asset.")

    # Upload zip asset
    file_bytes = zip_path.read_bytes()
    upload_target = f"{upload_base}?name={asset_name}"
    print(f"Uploading {asset_name} ({len(file_bytes) / (1024*1024):.1f} MB) to GitHub Release...")

    up_headers = {
        **headers,
        "Content-Type": "application/zip",
        "Content-Length": str(len(file_bytes)),
    }
    up_req = urllib.request.Request(upload_target, data=file_bytes, headers=up_headers, method="POST")

    with urllib.request.urlopen(up_req) as up_resp:
        result = json.loads(up_resp.read().decode())
        print(f"[SUCCESS] Uploaded asset: {result['name']} (ID: {result['id']})")
        print(f"Download URL: {result['browser_download_url']}")


if __name__ == "__main__":
    create_release_and_upload()
