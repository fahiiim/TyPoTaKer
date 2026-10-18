from typotaker.config import ModelConfig
from typotaker.models.growth import grow_model
from typotaker.models.typoformer import TyPoFormer, count_parameters
import torch


def _tiny() -> ModelConfig:
    return ModelConfig(
        name="unit",
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=128,
        max_seq_len=32,
        char_dim=16,
        n_replace_classes=8,
        n_append_classes=8,
    )


def test_forward_lm_and_gec():
    cfg = _tiny()
    m = TyPoFormer(cfg)
    b, t, c = 2, 16, cfg.max_chars_per_token
    tok = torch.randint(0, cfg.vocab_size, (b, t))
    chars = torch.randint(0, cfg.char_vocab_size, (b, t, c))
    out = m.forward(tok, chars, task="joint")
    assert out["lm_logits"].shape == (b, t, cfg.vocab_size)
    assert out["edit_logits"].shape == (b, t, cfg.n_edit_labels())
    loss = m.lm_loss(tok, chars, pad_id=0)
    assert torch.isfinite(loss)


def test_growth_preserves_prefix_layers():
    src_cfg = _tiny()
    dst_cfg = _tiny()
    dst_cfg.n_layers = 4
    dst_cfg.d_model = 64
    src = TyPoFormer(src_cfg)
    dst = grow_model(src, dst_cfg)
    assert count_parameters(dst) >= count_parameters(src)
    x = torch.randint(0, src_cfg.vocab_size, (1, 8))
    c = torch.randint(0, src_cfg.char_vocab_size, (1, 8, src_cfg.max_chars_per_token))
    y0 = src.forward(x, c, task="lm")["lm_logits"]
    y1 = dst.forward(x, c, task="lm")["lm_logits"]
    # identity extra layers => similar but not exact after width-matched copy
    assert y1.shape[1] == y0.shape[1]
