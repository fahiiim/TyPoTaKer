"""Grow a saved Stage-0 model into a larger Stage-1 config without losing weights."""

from __future__ import annotations

import torch
import torch.nn as nn

from typotaker.config import ModelConfig
from typotaker.models.typoformer import TyPoFormer, make_identity_block


def _copy_linear(dst: nn.Linear, src: nn.Linear) -> None:
    o_out, o_in = src.weight.shape
    dst.weight.data[:o_out, :o_in] = src.weight.data
    if src.bias is not None and dst.bias is not None:
        dst.bias.data[: src.bias.numel()] = src.bias.data


def _copy_embed(dst: nn.Embedding, src: nn.Embedding) -> None:
    rows = min(dst.weight.size(0), src.weight.size(0))
    cols = min(dst.weight.size(1), src.weight.size(1))
    dst.weight.data[:rows, :cols] = src.weight.data[:rows, :cols]


def grow_model(src: TyPoFormer, dst_cfg: ModelConfig) -> TyPoFormer:
    dst = TyPoFormer(dst_cfg)
    src_cfg = src.cfg
    if dst_cfg.vocab_size != src_cfg.vocab_size:
        raise ValueError("Growing across vocab sizes is not supported; retokenize first.")

    _copy_embed(dst.tok_embed, src.tok_embed)
    dst.char_enc.load_state_dict(
        _widen_block_state(src.char_enc.state_dict(), dst.char_enc.state_dict()),
        strict=False,
    )
    dst.char_gate.data.fill_(float(src.char_gate.detach()))
    for s_p, d_p in zip(src.mtp_proj, dst.mtp_proj):
        _copy_linear(d_p, s_p)

    n_copy = min(src_cfg.n_layers, dst_cfg.n_layers)
    for i in range(n_copy):
        s_b, d_b = src.blocks[i], dst.blocks[i]
        d_b.load_state_dict(_widen_block_state(s_b.state_dict(), d_b.state_dict()), strict=False)

    if dst_cfg.n_layers > src_cfg.n_layers:
        # replace extra layers with identity residuals using *destination* width
        for i in range(src_cfg.n_layers, dst_cfg.n_layers):
            dst.blocks[i] = make_identity_block(dst_cfg)

    _copy_linear(dst.edit_head, src.edit_head)
    if not dst_cfg.tie_embeddings:
        _copy_linear(dst.lm_head, src.lm_head)
    dst.norm.weight.data[: src.norm.weight.numel()] = src.norm.weight.data
    return dst


def _widen_block_state(src: dict, dst: dict) -> dict:
    out = {}
    for k, dv in dst.items():
        sv = src.get(k)
        if sv is None or sv.shape == dv.shape:
            out[k] = sv if sv is not None else dv
            continue
        cloned = dv.clone()
        slices = tuple(slice(0, min(a, b)) for a, b in zip(sv.shape, dv.shape))
        cloned[slices] = sv[slices]
        out[k] = cloned
    return out
