"""End-to-end automated pipeline for TyPoTaKer Stage 2 v4 training, export, and evaluation.

Orchestrates:
1. GPU hardware check (RTX 5070 12GB)
2. LM Stage 2 v4 training (10,000 steps on multi-domain text)
3. LM ONNX FP32 + INT8 export and validation
4. GEC Stage 2 v4 training (16,000 steps initialized from LM v4)
5. GEC ONNX FP32 + INT8 export and validation
6. Held-out benchmark evaluation on data/eval/gec_dev.jsonl
7. Desktop app publish to app/bin/publish
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

VENV_PY = sys.executable
DOTNET_EXE = r"C:\Users\Md Fahim Sarker\.dotnet\dotnet.exe"


def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def run_step(cmd: list[str], desc: str, check: bool = True) -> int:
    log(f"=== Starting: {desc} ===")
    log(f"Command: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0
    if check and res.returncode != 0:
        log(f"[ERROR] Step '{desc}' failed with exit code {res.returncode} after {elapsed:.1f}s")
        sys.exit(res.returncode)
    log(f"=== Completed: {desc} in {elapsed:.1f}s ({elapsed/60:.1f}m) ===")
    return res.returncode


def verify_env() -> None:
    log("Checking environment and GPU availability...")
    import torch
    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log(f"GPU Device: {name} ({vram_gb:.2f} GB VRAM)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    required_files = [
        "configs/train/lm_stage2_v4.yaml",
        "configs/train/gec_stage2_v4.yaml",
        "data/tokenizer/tokenizer.json",
        "data/processed/lm/train_v4.jsonl",
        "data/processed/gec/v4/train.jsonl",
        "data/processed/gec/v4/edit_maps.json",
    ]
    for rf in required_files:
        p = Path(rf)
        if not p.exists():
            log(f"[ERROR] Required file missing: {rf}")
            sys.exit(1)
        log(f"  OK: {rf} ({p.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="TyPoTaKer v4 End-to-End Pipeline")
    parser.add_argument("--skip-lm", action="store_true", help="Skip LM training phase")
    parser.add_argument("--skip-gec", action="store_true", help="Skip GEC training phase")
    parser.add_argument("--lm-only", action="store_true", help="Only run LM training and export")
    parser.add_argument("--gec-only", action="store_true", help="Only run GEC training and export")
    args = parser.parse_args()

    pipeline_t0 = time.time()
    log("=================================================================")
    log("     TYPOTAKER STAGE 2 v4 TRAINING & EXPORT PIPELINE             ")
    log("=================================================================")

    verify_env()

    # -------------------------------------------------------------
    # PHASE 1: LM Stage 2 v4 Training
    # -------------------------------------------------------------
    if not args.skip_lm and not args.gec_only:
        log("\n>>> PHASE 1: Training Language Model (LM Stage 2 v4, 10,000 steps)...")
        run_step(
            [VENV_PY, "-m", "typotaker", "train", "--config", "configs/train/lm_stage2_v4.yaml"],
            "LM Stage 2 v4 Training (10k steps)",
        )

        log("\n>>> PHASE 2: Exporting and Quantizing LM Stage 2 v4 ONNX models...")
        Path("export").mkdir(exist_ok=True)
        run_step(
            [
                VENV_PY,
                "-m",
                "typotaker",
                "export",
                "--run",
                "checkpoints/lm_stage2_v4",
                "--task",
                "lm",
                "--out",
                "export/lm_stage2.onnx",
                "--quantize",
                "--quantize-out",
                "export/lm_stage2_int8.onnx",
            ],
            "Export & Quantize LM Stage 2 v4 ONNX",
        )

        # Quick LM inference check
        run_step(
            [
                VENV_PY,
                "-m",
                "typotaker",
                "suggest",
                "--run",
                "checkpoints/lm_stage2_v4",
                "--text",
                "Please find the latest version of the",
                "-k",
                "5",
            ],
            "LM Autocomplete Sanity Check",
            check=False,
        )

    if args.lm_only:
        log("\n[INFO] --lm-only specified; exiting before GEC training.")
        return

    # -------------------------------------------------------------
    # PHASE 3: GEC Stage 2 v4 Training
    # -------------------------------------------------------------
    if not args.skip_gec:
        log("\n>>> PHASE 3: Training Grammar & Typo Corrector (GEC Stage 2 v4, 16,000 steps)...")
        run_step(
            [VENV_PY, "-m", "typotaker", "train", "--config", "configs/train/gec_stage2_v4.yaml"],
            "GEC Stage 2 v4 Training (16k steps)",
        )

        log("\n>>> PHASE 4: Exporting and Quantizing GEC Stage 2 v4 ONNX models...")
        run_step(
            [
                VENV_PY,
                "-m",
                "typotaker",
                "export",
                "--run",
                "checkpoints/gec_stage2_v4",
                "--task",
                "gec",
                "--out",
                "export/gec_stage2.onnx",
                "--quantize",
                "--quantize-out",
                "export/gec_stage2_int8.onnx",
            ],
            "Export & Quantize GEC Stage 2 v4 ONNX",
        )

        # Bundle v4 edit maps into export/
        shutil.copy("data/processed/gec/v4/edit_maps.json", "export/edit_maps.json")
        log("Copied v4 edit_maps.json -> export/edit_maps.json")

        # -------------------------------------------------------------
        # PHASE 5: Held-out Benchmark Evaluation
        # -------------------------------------------------------------
        log("\n>>> PHASE 5: Benchmarking on held-out 200-case dev benchmark...")
        run_step(
            [
                VENV_PY,
                "scripts/evaluate_gec.py",
                "--stage0",
                "checkpoints/gec_stage2",
                "--stage1",
                "checkpoints/gec_stage2_v4",
                "--cases",
                "data/eval/gec_dev.jsonl",
                "--maps",
                "data/processed/gec/v4/edit_maps.json",
                "--min-edit-prob",
                "0.85",
                "--edit-margin",
                "0.10",
            ],
            "Evaluate GEC Dev Benchmark",
            check=False,
        )

        # -------------------------------------------------------------
        # PHASE 6: Rebuild & Publish Desktop App
        # -------------------------------------------------------------
        log("\n>>> PHASE 6: Rebuilding and publishing TyPoTaKer desktop executable...")
        # Stop any running instances
        subprocess.run(["powershell", "-Command", "Stop-Process -Name 'TyPoTaKer.Desktop' -Force -ErrorAction SilentlyContinue"])
        time.sleep(1)

        dotnet = DOTNET_EXE if Path(DOTNET_EXE).exists() else "dotnet"
        run_step(
            [
                dotnet,
                "publish",
                "app/TyPoTaKer.Desktop/TyPoTaKer.Desktop.csproj",
                "-c",
                "Release",
                "-o",
                "app/bin/publish",
            ],
            "Publish Desktop Executable",
            check=False,
        )

    total_time = time.time() - pipeline_t0
    log("=================================================================")
    log(f"   PIPELINE COMPLETE in {total_time:.1f}s ({total_time/3600:.2f} hours)")
    log("=================================================================")


if __name__ == "__main__":
    main()
