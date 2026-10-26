import time
import torch
from typotaker.config import TrainConfig
from typotaker.tokenizer import TypoTokenizer
from typotaker.models.typoformer import TyPoFormer
from typotaker.data.dataset import GECDataset, collate
from torch.utils.data import DataLoader

cfg = TrainConfig.from_yaml("configs/train/gec_stage2_grammarly.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, torch.cuda.get_device_name(0))

tokenizer = TypoTokenizer(cfg.data.tokenizer_path, max_chars=cfg.model.max_chars_per_token)
model = TyPoFormer(cfg.model).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test batch
b, t = 64, 128
tok = torch.randint(0, 1000, (b, t), device=device)
char = torch.randint(0, 100, (b, t, 16), device=device)
labels = torch.randint(0, 5, (b, t), device=device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

# Warmup
for _ in range(5):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model.gec_loss(tok, char, labels, pad_id=0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

torch.cuda.synchronize()
t0 = time.perf_counter()
n_steps = 50
for _ in range(n_steps):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model.gec_loss(tok, char, labels, pad_id=0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()
dt = time.perf_counter() - t0

ms_per_step = (dt / n_steps) * 1000
tokens_per_sec = (n_steps * b * t) / dt
mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
mem_res = torch.cuda.max_memory_reserved() / (1024 ** 3)

print(f"Time per micro-batch (b={b}, t={t}): {ms_per_step:.2f} ms")
print(f"Throughput: {tokens_per_sec:,.0f} tokens/sec")
print(f"Peak VRAM Allocated: {mem_alloc:.2f} GB, Reserved: {mem_res:.2f} GB")
