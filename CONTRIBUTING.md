# Contributing to TyPoTaKer

## Layout

| Path | What lives there |
|---|---|
| `src/typotaker/models` | TyPoFormer, growth |
| `src/typotaker/train` | CUDA trainer + atomic checkpoints |
| `src/typotaker/data` | corpora, typo injection, edit alignment |
| `configs/` | stage and 6-hour train recipes |
| `tests/` | unit tests that run on CPU |

## Rules

1. Do not check in `checkpoints/`, `data/processed/`, or ONNX weights.
2. Every train run must be resumable from `checkpoints/<run>/latest.json`.
3. New architecture ideas belong in a new **stage** YAML, not a breaking rewrite of Stage 0.
4. Prefer precision on GEC (wrong edits are worse than missed edits).
5. Run `pytest` and `python -m typotaker count-params` before opening a PR.

## Training budget

Default recipes stop at **6 hours**. If you change `max_steps`, keep `max_hours` as the hard cap.
