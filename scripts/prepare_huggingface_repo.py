"""
Prepares a clean, standalone distribution folder for publishing TypoFormer to Hugging Face Hub.
Only packages model weights, tokenizers, ONNX graphs, configs, and a standalone runner.
Excludes all desktop/app, WPF, and installer code.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def prepare_hf_dist(output_dir: str = "hf_dist") -> Path:
    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parent.parent

    # 1. Model Weights (ONNX INT8 production weights)
    print("Copying ONNX models...")
    shutil.copy2(root / "export/lm_stage2_int8.onnx", out / "lm_stage2_int8.onnx")
    shutil.copy2(root / "export/gec_stage2_int8.onnx", out / "gec_stage2_int8.onnx")

    # 2. Tokenizer Files
    print("Copying tokenizers...")
    shutil.copy2(root / "data/tokenizer/vocab.json", out / "vocab.json")
    shutil.copy2(root / "data/tokenizer/merges.txt", out / "merges.txt")
    shutil.copy2(root / "data/tokenizer/tokenizer.json", out / "tokenizer.json")

    # 3. GEC Confusion Maps
    print("Copying edit maps...")
    shutil.copy2(root / "data/processed/gec/v4/edit_maps.json", out / "edit_maps.json")

    # 4. Model Config JSON
    print("Generating config.json...")
    config = {
        "model_type": "typoformer",
        "architectures": ["TypoFormerForCausalLMAndGEC"],
        "vocab_size": 16000,
        "max_seq_len": 512,
        "max_chars_per_token": 16,
        "hidden_dim": 768,
        "n_blocks": 10,
        "n_q_heads": 12,
        "n_kv_heads": 4,
        "ffn_dim": 2048,
        "char_cnn_filters": 64,
        "char_cnn_kernels": [2, 3, 5, 8],
        "n_classes": 1024,
        "int8_quantized": True,
        "framework": "onnxruntime",
        "parameters": 93888001
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # 5. Tokenizer Config JSON
    tok_config = {
        "tokenizer_class": "TypoTokenizer",
        "model_type": "bpe",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "max_chars_per_token": 16
    }
    (out / "tokenizer_config.json").write_text(json.dumps(tok_config, indent=2), encoding="utf-8")

    # 6. Standalone Python Inference Script
    print("Generating standalone infer.py...")
    infer_code = '''"""
TypoFormer: Standalone Python Inference Runner
Supports both Autocomplete (Causal LM) and Grammar Correction (GEC).

Requirements:
    pip install onnxruntime tokenizers numpy
"""

import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

class TypoFormer:
    def __init__(self, model_dir: str | None = None):
        p = Path(__file__).resolve().parent if model_dir is None else Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(p / "tokenizer.json"))
        
        # Load ONNX sessions
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        
        self.lm_session = ort.InferenceSession(str(p / "lm_stage2_int8.onnx"), opts)
        self.gec_session = ort.InferenceSession(str(p / "gec_stage2_int8.onnx"), opts)
        
        # Load confusion maps
        maps = json.loads((p / "edit_maps.json").read_text(encoding="utf-8"))
        self.id_to_replace = {int(k): int(v) for k, v in maps["id_to_replace"].items()}
        self.id_to_append = {int(k): int(v) for k, v in maps["id_to_append"].items()}

    def _prepare_inputs(self, text: str):
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids
        tokens = encoding.tokens

        # Character IDs (UTF-8 bytes capped at 16, mapped to 0..127 ASCII)
        char_matrix = []
        for t in tokens:
            raw = t.encode("utf-8", errors="replace")[:16]
            b = [byte if 0 < byte < 128 else 127 for byte in raw]
            b += [0] * (16 - len(b))
            char_matrix.append(b)
            
        token_tensor = np.array([token_ids], dtype=np.int64)
        char_tensor = np.array([char_matrix], dtype=np.int64)
        return token_tensor, char_tensor, tokens

    def autocomplete(self, text: str, top_k: int = 3):
        token_tensor, char_tensor, _ = self._prepare_inputs(text)
        outputs = self.lm_session.run(None, {
            "token_ids": token_tensor,
            "char_ids": char_tensor
        })
        logits = outputs[0][0, -1] # Last token prediction
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)
        
        top_indices = np.argsort(probs)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            token = self.tokenizer.decode([idx])
            results.append((token, float(probs[idx])))
        return results

    def correct_grammar(self, text: str):
        token_tensor, char_tensor, tokens = self._prepare_inputs(text)
        outputs = self.gec_session.run(None, {
            "token_ids": token_tensor,
            "char_ids": char_tensor
        })
        logits = outputs[0][0] # (seq_len, n_classes)
        actions = np.argmax(logits, axis=-1)
        
        corrected = []
        for i, (tok, act) in enumerate(zip(tokens, actions)):
            if act == 0: # KEEP
                corrected.append(tok)
            elif act == 1: # DELETE
                continue
            elif act == 2: # CASE
                corrected.append(tok.capitalize() if tok.islower() else tok.lower())
            elif 5 <= act < 5 + len(self.id_to_replace): # REPLACE
                repl_id = self.id_to_replace.get(act - 5)
                corrected.append(self.tokenizer.decode([repl_id]) if repl_id else tok)
            else:
                corrected.append(tok)
                
        return "".join(corrected).replace("Ġ", " ").strip()


if __name__ == "__main__":
    model = TypoFormer()
    
    # 1. Test Autocomplete
    print("--- Autocomplete Test ---")
    prefix = "Could you please review the configuration file before the "
    print(f"Prefix: {prefix}")
    print("Suggestions:", model.autocomplete(prefix, top_k=3))
    
    # 2. Test Grammar & Typo Fix
    print("\\n--- Grammar Correction Test ---")
    dirty = "This is teh wrong way to write."
    print(f"Original:  {dirty}")
    print(f"Corrected: {model.correct_grammar(dirty)}")
'''
    (out / "infer.py").write_text(infer_code.strip() + "\n", encoding="utf-8")

    # 7. Hugging Face Model Card README.md
    print("Generating Hugging Face Model Card README.md...")
    hf_readme = """---
language:
- en
license: mit
tags:
- onnx
- causal-lm
- grammar-correction
- typo-correction
- character-aware
- rope
- gqa
- swiglu
- int8
- offline-nlp
pipeline_tag: text2text-generation
inference: false
---

# TypoFormer (Stage 2 INT8)

**TypoFormer** is a 100% offline, character-aware hybrid Transformer designed for ultra-low latency (<10ms on CPU) next-word autocomplete and non-autoregressive grammatical error correction (GEC).

## Model Architecture
- **Trunk**: 10 Transformer blocks (768 hidden dimension, 2048 FFN dimension)
- **Attention**: Grouped Query Attention (12 Query heads, 4 Key-Value heads) + Rotary Position Embeddings (RoPE)
- **Non-Linearity**: SwiGLU + Pre-RMSNorm
- **Hybrid Input**: Gated fusion of 16,000 Byte-Level BPE embeddings with Multi-Scale Character 1D-Convolutions (kernel sizes 2, 3, 5, 8)
- **Quantization**: Symmetric INT8 ONNX graph (130 MB footprint)

## Files in this Repository
- `lm_stage2_int8.onnx`: Causal language model for next-word suggestion (~138 MB)
- `gec_stage2_int8.onnx`: Non-autoregressive edit-tagger for grammar & typo correction (~130 MB)
- `tokenizer.json`, `vocab.json`, `merges.txt`: 16k Byte-Level BPE tokenizer
- `edit_maps.json`: High-frequency confusion dictionary for atomic replacements & appends
- `infer.py`: Self-contained Python runner using ONNX Runtime

## Quickstart

```bash
pip install onnxruntime tokenizers numpy
git clone https://huggingface.co/<your-username>/TypoFormer
cd TypoFormer
python infer.py
```

### Python Example
```python
from infer import TypoFormer

model = TypoFormer(".")

# Autocomplete
suggestions = model.autocomplete("Here is the ", top_k=3)
print("Autocomplete:", suggestions)

# Grammar & Typo Fix
clean = model.correct_grammar("This is teh first test.")
print("Fixed:", clean)
```

## License
MIT
"""
    (out / "README.md").write_text(hf_readme.strip() + "\n", encoding="utf-8")

    print(f"\n[OK] Hugging Face distribution package prepared at: {out.resolve()}")
    return out


if __name__ == "__main__":
    prepare_hf_dist()
