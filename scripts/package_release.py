"""
Packages the clean, standalone release archive for TypoFormer.
Produces:
1. installer/payload/ (uncompressed ready-to-use directory)
2. installer/TypoFormer-win-x64.zip (compressed distribution zip)
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path


def package_release():
    root = Path(__file__).resolve().parent.parent
    publish_dir = root / "app" / "bin" / "publish"
    payload_dir = root / "installer" / "payload"
    zip_path = root / "installer" / "TypoFormer-win-x64.zip"

    if not (publish_dir / "TyPoTaKer.Desktop.exe").exists():
        raise FileNotFoundError(f"Missing publish binaries at {publish_dir}")

    print("Cleaning payload directory...")
    if payload_dir.exists():
        shutil.rmtree(payload_dir)
    payload_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy essential binaries and configs
    print("Copying core executable and DLLs...")
    for ext in ("*.dll", "*.exe", "*.json", "*.ico", "*.png"):
        for f in publish_dir.glob(ext):
            shutil.copy2(f, payload_dir / f.name)

    # 2. Copy native runtimes (win-x64)
    runtimes_win = publish_dir / "runtimes" / "win-x64"
    if runtimes_win.exists():
        target_runtimes = payload_dir / "runtimes" / "win-x64"
        shutil.copytree(runtimes_win, target_runtimes)

    # 3. Copy only quantized INT8 models and maps (skip bulky FP32)
    print("Copying Stage 2 INT8 models and confusion maps...")
    (payload_dir / "export").mkdir(parents=True, exist_ok=True)
    for model_name in ("lm_stage2_int8.onnx", "gec_stage2_int8.onnx", "edit_maps.json"):
        src = publish_dir / "export" / model_name
        if src.exists():
            shutil.copy2(src, payload_dir / "export" / model_name)

    # 4. Copy tokenizer & dataset maps
    print("Copying tokenizers...")
    (payload_dir / "data" / "tokenizer").mkdir(parents=True, exist_ok=True)
    for tok_file in (publish_dir / "data" / "tokenizer").glob("*"):
        shutil.copy2(tok_file, payload_dir / "data" / "tokenizer" / tok_file.name)

    (payload_dir / "data" / "processed" / "gec" / "v4").mkdir(parents=True, exist_ok=True)
    v4_map = publish_dir / "data" / "processed" / "gec" / "v4" / "edit_maps.json"
    if v4_map.exists():
        shutil.copy2(v4_map, payload_dir / "data" / "processed" / "gec" / "v4" / "edit_maps.json")

    # Calculate payload size
    total_bytes = sum(f.stat().st_size for f in payload_dir.rglob("*") if f.is_file())
    mb = total_bytes / (1024 * 1024)
    print(f"[OK] Clean payload prepared at {payload_dir} ({mb:.1f} MB)")

    # 5. Build compressed distribution zip
    print(f"Creating compressed distribution archive: {zip_path.name}...")
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in payload_dir.rglob("*"):
            if f.is_file():
                rel_path = f.relative_to(payload_dir)
                zf.write(f, rel_path)

    zip_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Created {zip_path.name} ({zip_mb:.1f} MB)")


if __name__ == "__main__":
    package_release()
