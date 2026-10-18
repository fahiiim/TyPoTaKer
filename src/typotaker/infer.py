from __future__ import annotations

import json
from pathlib import Path

import torch

from typotaker.config import ModelConfig
from typotaker.data.edits import CASE_CAP, CASE_LOWER, DELETE, KEEP, N_BASIC
from typotaker.models.typoformer import TyPoFormer
from typotaker.tokenizer import TypoTokenizer
from typotaker.train.checkpoint import find_latest, load_checkpoint


def load_run(run_path: str, tokenizer_path: str, device: str | None = None) -> tuple[TyPoFormer, TypoTokenizer]:
    path = Path(run_path)
    ckpt = find_latest(path) if path.is_dir() else path
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint under {run_path}")
    blob = load_checkpoint(ckpt)  # type: ignore[arg-type]
    cfg = ModelConfig(**blob["meta"]["model"])
    model = TyPoFormer(cfg)
    model.load_state_dict(blob["model"])
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device_t).eval()
    tok = TypoTokenizer(tokenizer_path, max_chars=cfg.max_chars_per_token)
    return model, tok


@torch.no_grad()
def next_words(model: TyPoFormer, tok: TypoTokenizer, text: str, k: int = 5) -> list[tuple[str, float]]:
    ids = tok.encode(text, add_special=True)[:-1]  # drop eos, keep bos+tokens
    pieces = tok.token_strings(text, add_special=True)[:-1]
    chars = tok.char_ids_for_tokens(pieces)
    device = next(model.parameters()).device
    token_ids = torch.tensor([ids], device=device)
    char_ids = torch.tensor([chars], device=device)
    logits = model.forward(token_ids, char_ids, task="lm")["lm_logits"][0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    val, idx = torch.topk(probs, k)
    out = []
    for p, i in zip(val.tolist(), idx.tolist()):
        out.append((tok.decode([i]), p))
    return out


def _invert_maps(maps_path: str) -> tuple[dict[int, int], dict[int, int]]:
    raw = json.loads(Path(maps_path).read_text(encoding="utf-8"))
    id_to_replace = {int(k): int(v) for k, v in raw["id_to_replace"].items()}
    id_to_append = {int(k): int(v) for k, v in raw["id_to_append"].items()}
    return id_to_replace, id_to_append


@torch.no_grad()
def correct_sentence(
    model: TyPoFormer,
    tok: TypoTokenizer,
    text: str,
    maps_path: str,
) -> str:
    id_to_replace, id_to_append = _invert_maps(maps_path)
    ids = tok.encode(text, add_special=True)
    pieces = tok.token_strings(text, add_special=True)
    chars = tok.char_ids_for_tokens(pieces)
    device = next(model.parameters()).device
    logits = model.forward(
        torch.tensor([ids], device=device),
        torch.tensor([chars], device=device),
        task="gec",
        pad_id=tok.pad_id,
    )["edit_logits"][0]
    labels = logits.argmax(dim=-1).tolist()
    n_replace = model.cfg.n_replace_classes
    new_ids: list[int] = []
    for tid, lab, piece in zip(ids, labels, pieces):
        if lab == DELETE:
            continue
        token = tid
        if lab == CASE_CAP:
            decoded = tok.decode([tid]).capitalize()
            rec = tok.encode(decoded, add_special=False)
            token = rec[0] if rec else tid
        elif lab == CASE_LOWER:
            decoded = tok.decode([tid]).lower()
            rec = tok.encode(decoded, add_special=False)
            token = rec[0] if rec else tid
        elif N_BASIC <= lab < N_BASIC + n_replace:
            token = id_to_replace.get(lab, tid)
        new_ids.append(token)
        if lab >= N_BASIC + n_replace:
            extra = id_to_append.get(lab)
            if extra is not None:
                new_ids.append(extra)
        _ = piece, KEEP
    return tok.decode(new_ids).strip()
