import onnxruntime as ort
import json
from tokenizers import Tokenizer
import numpy as np

from typotaker.tokenizer import TypoTokenizer
tok = TypoTokenizer('data/tokenizer/tokenizer.json')
lm_sess = ort.InferenceSession('export/lm_stage1_int8.onnx')

roman_numerals = {'iv', 'ix', 'vi', 'vii', 'viii', 'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'ii', 'iii'}

def get_autocomplete(text, top_k=3):
    trimmed = text.strip()
    if not trimmed:
        return []
    tokens = [tok.bos_id] + tok.encode(trimmed, add_special=False)
    pieces = ['<bos>'] + tok.token_strings(trimmed, add_special=False)
    char_rows = tok.char_ids_for_tokens(pieces)
    char_ids = np.array([char_rows], dtype=np.int64)
    tok_ids = np.array([tokens], dtype=np.int64)
    
    out = lm_sess.run(None, {'token_ids': tok_ids, 'char_ids': char_ids})[0]
    logits = out[0, -1]
    probs = np.exp(logits - np.max(logits))
    probs /= np.sum(probs)
    
    sorted_idx = np.argsort(probs)[::-1]
    results = []
    
    for idx in sorted_idx:
        p = float(probs[idx])
        if p < 0.005:
            break
        raw_tok = tok.tok.id_to_token(int(idx))
        decoded = tok.decode([int(idx)]).strip()
        
        # Must start with Ġ (a new word)
        if not raw_tok.startswith('Ġ'):
            continue
        # Clean letters only
        if not decoded.isalpha():
            continue
        # Length check
        if len(decoded) == 1 and decoded.lower() not in ['a', 'i']:
            continue
        # Roman numerals filter
        if decoded.lower() in roman_numerals:
            continue
            
        results.append((decoded, p))
        if len(results) >= top_k:
            break
            
    return results

for test_input in ['put your head on', 'put your head on ', 'put your head on my', 'put your head on my ']:
    suggestions = get_autocomplete(test_input)
    print(f'Input: {repr(test_input)} -> Suggestions: {suggestions}')
