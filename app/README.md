# Desktop app (next slice)

The training stack in `src/typotaker` is the open-source core. The always-on Windows overlay will live here.

## Target shape

- Tray process + tiny overlay near the caret
- ONNX Runtime (int8) for both heads: next-word and edit tags
- UI Automation to read the focused text field; keyboard hook as fallback
- Hotkey to accept (`Ctrl+Space`); never intercept Tab
- Skip password fields and a per-app blocklist
- Offline only

## Why this folder is a stub

The model, checkpoints, tokenizer, and export path need to exist first. After a Stage 0 GEC run:

```bash
python -m typotaker export --run checkpoints/gec_stage0 --task gec --out app/models/gec.onnx
python -m typotaker export --run checkpoints/lm_stage0 --task lm --out app/models/lm.onnx
```

Then we can put a Tauri/Rust (`ort`) or C# / ONNX Runtime shell in this directory without moving training code.

A Python tray app is intentionally **not** the shipping path: it is too heavy for “lightweight OS-wide typing”.
