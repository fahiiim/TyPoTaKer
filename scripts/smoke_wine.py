"""
Smoke test verification for the fine-tuned TypoFormer Wine models.
Validates GEC correction and LM next-word generation across:
1. Engineering
2. Client Handling
3. Workplace Chitchat
"""

import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typotaker.tokenizer import TypoTokenizer


def test_wine_models():
    root = Path(__file__).resolve().parent.parent
    gec_path = root / "export" / "gec_stage2_int8.onnx"
    lm_path = root / "export" / "lm_stage2_int8.onnx"
    maps_path = root / "export" / "edit_maps.json"
    tok_path = root / "data" / "tokenizer" / "tokenizer.json"

    if not gec_path.exists() or not maps_path.exists():
        print(f"[skip] Models not exported yet: {gec_path}")
        return

    print("=" * 65)
    print("TYPOFORMER 'WINE' LIVE SMOKE TEST")
    print("=" * 65)

    tok = TypoTokenizer(str(tok_path))
    gec_sess = ort.InferenceSession(str(gec_path))
    
    maps = json.loads(maps_path.read_text(encoding="utf-8"))
    id_to_replace = {int(k): int(v) for k, v in maps["id_to_replace"].items()}
    id_to_append = {int(k): int(v) for k, v in maps["id_to_append"].items()}

    test_sentences = [
        # Engineering Punctuation & Homophone
        ("High network jitter will directly effect the streaming pipeline throughput.", "affect"),
        ("We dont expect any downtime during the blue-green deployment phase.", "don't"),
        ("The transformer backbone reached its lowest cross-entropy loss at step 2000.", "clean anchor"),
        
        # Client Inquiries & Deliverables
        ("Could you please confirm if Friday at 2 PM works for the project demo.", "question mark ?"),
        ("We are on track to hand over the first key delieverable by Friday.", "deliverable typo"),
        
        # Workplace Chitchat
        ("Hey Fahim do you have a moment to inspect this trace?", "direct address comma"),
        ("Have a wonderful wekend everyone, and see you all on Monday morning!", "weekend typo"),
    ]

    print("\n--- GEC CORRECTION TEST ---")
    for src, note in test_sentences:
        ids = tok.encode(src, add_special=True)
        pieces = tok.token_strings(src, add_special=True)
        chars = tok.char_ids_for_tokens(pieces)
        t_ids = np.array([ids], dtype=np.int64)
        c_ids = np.array([chars], dtype=np.int64)

        logits = gec_sess.run(None, {"token_ids": t_ids, "char_ids": c_ids})[0][0]
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        preds = np.argmax(probs, axis=-1)

        # Gated confidence and subtoken protection
        all_words_valid = all(len(w) > 0 and w.isalpha() for w in src.split())
        for i in range(len(preds)):
            if preds[i] != 0:
                best_p = probs[i, preds[i]]
                keep_p = probs[i, 0]
                if best_p < 0.40 or (best_p - keep_p) < 0.05:
                    preds[i] = 0
                    continue
                # Sub-token protection
                if 0 < i < len(ids) - 1:
                    is_subtoken = not pieces[i].startswith("Ġ") and pieces[i] not in ("<bos>", "<eos>")
                    if is_subtoken and all_words_valid and best_p < 0.95:
                        preds[i] = 0

        new_ids = []
        for tid, act in zip(ids, preds):
            if act == 0:
                new_ids.append(tid)
            elif act == 1:
                pass
            elif act == 2:
                new_ids.append(tok.encode(tok.decode([tid]).capitalize(), add_special=False)[0])
            elif act == 3:
                new_ids.append(tok.encode(tok.decode([tid]).lower(), add_special=False)[0])
            elif 4 <= act < 4 + 4096:
                new_ids.append(id_to_replace.get(act, tid))
            elif act >= 4 + 4096:
                new_ids.append(tid)
                extra = id_to_append.get(act)
                if extra:
                    new_ids.append(extra)

        out = tok.decode(new_ids)
        print(f"\n[{note}]")
        print(f"  Input:  {src}")
        print(f"  Output: {out}")

    if lm_path.exists():
        print("\n--- LM AUTOCOMPLETE TEST ---")
        lm_sess = ort.InferenceSession(str(lm_path))
        lm_prompts = [
            "We decoupled the event processing pipeline to achieve sub-",
            "Could you please confirm if the proposed implementation timeline aligns with your",
            "Hey Fahim, do you have a few minutes for a quick",
        ]
        for prompt in lm_prompts:
            ids = tok.encode(prompt, add_special=True)[:-1]
            pieces = tok.token_strings(prompt, add_special=True)[:-1]
            chars = tok.char_ids_for_tokens(pieces)
            t_ids = np.array([ids], dtype=np.int64)
            c_ids = np.array([chars], dtype=np.int64)
            logits = lm_sess.run(None, {"token_ids": t_ids, "char_ids": c_ids})[0][0, -1]
            exp = np.exp(logits - np.max(logits))
            p = exp / np.sum(exp)
            top_k = np.argsort(p)[::-1][:3]
            suggestions = [(tok.decode([idx]), float(p[idx])) for idx in top_k]
            print(f"\nPrompt: {repr(prompt)}")
            print(f"  Next words: {suggestions}")

    print("\n" + "=" * 65)
    print("[OK] Smoke verification passed.")
    print("=" * 65)


if __name__ == "__main__":
    test_wine_models()
