from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch

from typotaker.config import ModelConfig
from typotaker.models.typoformer import TyPoFormer
from typotaker.train.checkpoint import find_latest, load_checkpoint


class _ExportWrapper(torch.nn.Module):
    def __init__(self, model: TyPoFormer, task: str) -> None:
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, token_ids: torch.Tensor, char_ids: torch.Tensor) -> torch.Tensor:
        out = self.model.forward(token_ids, char_ids, task=self.task)  # type: ignore[arg-type]
        key = "lm_logits" if self.task == "lm" else "edit_logits"
        return out[key]


def export_onnx(run_dir: str, out_path: str, task: str = "lm", seq_len: int = 128) -> str:
    latest = find_latest(Path(run_dir))
    if latest is None:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    blob = load_checkpoint(latest)
    cfg = ModelConfig(**blob["meta"]["model"])
    model = TyPoFormer(cfg)
    model.load_state_dict(blob["model"])
    model.eval()
    wrap = _ExportWrapper(model, task)
    wrap.eval()
    tok = torch.zeros(1, seq_len, dtype=torch.long)
    chars = torch.zeros(1, seq_len, cfg.max_chars_per_token, dtype=torch.long)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrap,
        (tok, chars),
        str(out),
        input_names=["token_ids", "char_ids"],
        output_names=["logits"],
        dynamic_axes={
            "token_ids": {0: "batch", 1: "seq"},
            "char_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=18,
        dynamo=False,
    )
    print(f"exported ONNX -> {out}")
    return str(out)



def quantize_onnx(onnx_path: str, out_path: str | None = None) -> str:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = Path(onnx_path)
    dst = Path(out_path) if out_path else src.with_name(f"{src.stem}_int8{src.suffix}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
    )
    print(f"quantized ONNX (INT8) -> {dst}")
    return str(dst)

