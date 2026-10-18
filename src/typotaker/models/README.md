# TyPoFormer

Stage 0 target: **~25.4M** parameters (`configs/model/stage0_25m.yaml`).

Training-only **multi-token prediction** uses two `d×d` adapters and the tied LM head (t+2, t+3). They add ~0.3M weights and are unused at inference.

| Block | Role |
|---|---|
| `CharCNN` | 3×1-d conv over up to 16 bytes of each BPE piece |
| `char_gate` | learned scalar; starts small so LM can ignore chars until they help |
| `Attention` | GQA + RoPE + `scaled_dot_product_attention` (8 layers at Stage 0) |
| `FeedForward` | SwiGLU, no bias |
| `lm_head` | tied to token embedding |
| `edit_head` | 4 + 2048 + 2048 labels |

Depth growth (`growth.py`) copies existing blocks and **zeros** the output projections of new blocks so the residual path is identity at the moment of growth.
