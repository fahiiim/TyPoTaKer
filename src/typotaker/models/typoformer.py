"""TyPoFormer: character-aware tiny transformer with dual LM + edit heads.

Novelty that is still cheap:
- A gated character CNN is fused into each BPE token (typos live below word pieces).
- QK-RMSNorm + GQA + RoPE + SwiGLU for stable 12 GB CUDA throughput.
- Training-only multi-token prediction via tiny d×d adapters (shared tied LM head).
- Compact edit tags (KEEP/DELETE/CASE + top-k REPLACE/APPEND) instead of seq2seq.
- Depth/width growth with identity-initialized new layers so Stage 0 can become Stage 1.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from typotaker.config import ModelConfig

Task = Literal["lm", "gec", "joint"]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D], cos/sin: [T, D/2] broadcast via repeat
    cos = torch.cat([cos, cos], dim=-1)[None, None, :, :]
    sin = torch.cat([sin, sin], dim=-1)[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


class CharCNN(nn.Module):
    """Per-token character encoder. Tiny on purpose: typos are local."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.char_vocab_size, cfg.char_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(cfg.char_dim, cfg.char_dim * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cfg.char_dim * 2, cfg.char_dim * 2, kernel_size=3, padding=1)
        self.proj = nn.Linear(cfg.char_dim * 2, cfg.d_model, bias=False)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: [B, T, C]
        b, t, c = char_ids.shape
        x = self.embed(char_ids).view(b * t, c, -1).transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.amax(dim=-1)
        return self.proj(x).view(b, t, -1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must divide n_heads")
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.q_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal: bool,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        attn_mask = None
        if key_padding_mask is not None:
            # True = keep; SDPA wants True = mask out
            attn_mask = ~key_padding_mask
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None if attn_mask is None else attn_mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.wo(y)


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal: bool,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.attn(self.n1(x), cos, sin, causal, key_padding_mask)
        x = x + self.ffn(self.n2(x))
        return x


class TyPoFormer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.char_enc = CharCNN(cfg)
        self.char_gate = nn.Parameter(torch.tensor(cfg.char_gate_init))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embed.weight
        self.edit_head = nn.Linear(cfg.d_model, cfg.n_edit_labels(), bias=False)
        self.mtp_proj = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.d_model, bias=False) for _ in range(max(0, cfg.n_mtp))]
        )
        head_dim = cfg.d_model // cfg.n_heads
        self.rope = RotaryEmbedding(head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def encode(
        self,
        token_ids: torch.Tensor,
        char_ids: torch.Tensor,
        causal: bool,
        pad_id: int | None = None,
    ) -> torch.Tensor:
        b, t = token_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence {t} > max_seq_len {self.cfg.max_seq_len}")
        x = self.tok_embed(token_ids) + self.char_gate * self.char_enc(char_ids)
        cos, sin = self.rope(x, t)
        key_padding_mask = None if pad_id is None else token_ids.ne(pad_id)
        for block in self.blocks:
            x = block(x, cos, sin, causal=causal, key_padding_mask=key_padding_mask)
        return self.norm(x)

    def forward(
        self,
        token_ids: torch.Tensor,
        char_ids: torch.Tensor,
        task: Task = "lm",
        pad_id: int | None = None,
    ) -> dict[str, torch.Tensor]:
        causal = task in {"lm", "joint"}
        h = self.encode(token_ids, char_ids, causal=causal, pad_id=pad_id)
        out: dict[str, torch.Tensor] = {"hidden": h}
        if task in {"lm", "joint"}:
            out["lm_logits"] = self.lm_head(h)
        if task in {"gec", "joint"}:
            out["edit_logits"] = self.edit_head(h)
        return out

    def lm_loss(
        self,
        token_ids: torch.Tensor,
        char_ids: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor:
        inp_tok = token_ids[:, :-1]
        inp_char = char_ids[:, :-1]
        h = self.encode(inp_tok, inp_char, causal=True, pad_id=None)
        logits = self.lm_head(h)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            token_ids[:, 1:].reshape(-1),
            ignore_index=pad_id,
        )
        mtp_scales = (0.3, 0.15)
        for i, proj in enumerate(self.mtp_proj):
            offset = i + 2
            if inp_tok.size(1) <= offset:
                break
            extra = self.lm_head(proj(h[:, : -(offset - 1)]))
            tgt = token_ids[:, offset:]
            scale = mtp_scales[i] if i < len(mtp_scales) else 0.1
            loss = loss + scale * F.cross_entropy(
                extra.reshape(-1, extra.size(-1)),
                tgt.reshape(-1),
                ignore_index=pad_id,
            )
        return loss

    def gec_loss(
        self,
        token_ids: torch.Tensor,
        char_ids: torch.Tensor,
        edit_labels: torch.Tensor,
        pad_id: int,
        keep_id: int = 0,
        keep_downweight: float = 0.15,
    ) -> torch.Tensor:
        out = self.forward(token_ids, char_ids, task="gec", pad_id=pad_id)
        logits = out["edit_logits"]
        weight = torch.ones(logits.size(-1), device=logits.device)
        weight[keep_id] = keep_downweight
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            edit_labels.reshape(-1),
            weight=weight,
            ignore_index=-100,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_identity_block(cfg: ModelConfig) -> Block:
    """New residual block starts as near-identity so growth does not destroy Stage 0."""
    block = Block(cfg)
    nn.init.zeros_(block.attn.wo.weight)
    nn.init.zeros_(block.ffn.w2.weight)
    return block
