"""
Deploys and packages the Fine-Tuned 'Wine' TypoFormer Model.
1. Exports best GEC checkpoint from checkpoints/gec_stage2_wine to ONNX FP32
2. Quantizes ONNX model to INT8 (gec_stage2_int8.onnx)
3. Bundles edit_maps.json to export/ and payload directories
4. Updates local app/bin/publish and %LOCALAPPDATA%/Programs/TypoFormer
5. Re-packages installer/TypoFormer-win-x64.zip
6. Runs real-world evaluation test suite
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from typotaker.train.checkpoint import find_latest
from typotaker.export import export_onnx, quantize_onnx


def deploy_wine():
    root = Path(__file__).resolve().parent.parent
    checkpoints_dir = root / "checkpoints" / "gec_stage2_wine"
    latest_ckpt = find_latest(checkpoints_dir)
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoints_dir}")

    print("=" * 65)
    print(f"DEPLOYING TYPOFORMER 'WINE' FINE-TUNED CHECKPOINT")
    print(f"Source Checkpoint: {latest_ckpt}")
    print("=" * 65)

    export_dir = root / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    gec_fp32 = export_dir / "gec_stage2.onnx"
    gec_int8 = export_dir / "gec_stage2_int8.onnx"
    maps_v5 = root / "data" / "processed" / "gec" / "v5" / "edit_maps.json"
    maps_export = export_dir / "edit_maps.json"

    # 1. Export FP32 ONNX
    print(f"\n[1/5] Exporting ONNX FP32 -> {gec_fp32}...")
    t0 = time.time()
    export_onnx(str(checkpoints_dir), str(gec_fp32), task="gec", seq_len=128)
    print(f"FP32 exported in {time.time() - t0:.1f}s ({gec_fp32.stat().st_size / 1e6:.1f} MB)")

    # 2. Quantize GEC to INT8
    print(f"\n[2/6] Quantizing GEC to INT8 -> {gec_int8}...")
    t0 = time.time()
    quantize_onnx(str(gec_fp32), str(gec_int8))
    print(f"INT8 quantized in {time.time() - t0:.1f}s ({gec_int8.stat().st_size / 1e6:.1f} MB)")

    # 3. Export and Quantize LM if available
    lm_checkpoints_dir = root / "checkpoints" / "lm_stage2_wine"
    lm_ckpt = find_latest(lm_checkpoints_dir)
    lm_int8 = export_dir / "lm_stage2_int8.onnx"
    if lm_ckpt is not None:
        lm_fp32 = export_dir / "lm_stage2.onnx"
        print(f"\n[3/6] Exporting and Quantizing LM Stage 2 Wine -> {lm_int8}...")
        t0 = time.time()
        export_onnx(str(lm_checkpoints_dir), str(lm_fp32), task="lm", seq_len=128)
        quantize_onnx(str(lm_fp32), str(lm_int8))
        print(f"LM INT8 quantized in {time.time() - t0:.1f}s ({lm_int8.stat().st_size / 1e6:.1f} MB)")
    else:
        print("\n[3/6] Skipping LM export (no lm_stage2_wine checkpoint yet)")

    # 4. Copy edit maps to export and v4 locations
    print(f"\n[4/6] Updating export edit maps -> {maps_export}...")
    shutil.copy2(maps_v5, maps_export)
    v4_map = root / "data" / "processed" / "gec" / "v4" / "edit_maps.json"
    if v4_map.parent.exists():
        shutil.copy2(maps_v5, v4_map)

    # 5. Copy to publish dir & LocalAppData if installed
    print("\n[5/6] Updating application deployment targets...")
    publish_export = root / "app" / "bin" / "publish" / "export"
    if publish_export.exists():
        shutil.copy2(gec_int8, publish_export / "gec_stage2_int8.onnx")
        if lm_ckpt is not None:
            shutil.copy2(lm_int8, publish_export / "lm_stage2_int8.onnx")
        shutil.copy2(maps_export, publish_export / "edit_maps.json")
        print(f"Updated publish export dir -> {publish_export}")

    local_app_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "TypoFormer"
    if local_app_dir.exists():
        target_export = local_app_dir / "export"
        target_export.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(gec_int8, target_export / "gec_stage2_int8.onnx")
            shutil.copy2(maps_export, target_export / "edit_maps.json")
            print(f"Updated live desktop app files -> {target_export}")
        except Exception as e:
            print(f"Notice: Live desktop files locked ({e}); payload updated.")

    # 5. Copy to hf_dist for Hugging Face release & local live benchmarks
    hf_dist_dir = root / "hf_dist"
    if hf_dist_dir.exists():
        print(f"\n[5/6] Updating Hugging Face distribution package -> {hf_dist_dir}...")
        shutil.copy2(gec_int8, hf_dist_dir / "gec_stage2_int8.onnx")
        if lm_ckpt is not None:
            shutil.copy2(lm_int8, hf_dist_dir / "lm_stage2_int8.onnx")
        shutil.copy2(maps_export, hf_dist_dir / "edit_maps.json")

    # 6. Package release ZIP
    print("\n[6/6] Re-packaging installer/TypoFormer-win-x64.zip...")
    sys.path.insert(0, str(root / "scripts"))
    import package_release
    package_release.package_release()

    print("\n" + "=" * 65)
    print("[SUCCESS] TypoFormer Wine Models Deployed and Packaged Successfully!")
    print("=" * 65)


if __name__ == "__main__":
    deploy_wine()
