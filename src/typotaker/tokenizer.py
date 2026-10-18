from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL = [PAD, UNK, BOS, EOS]


def train_bpe(corpus_files: list[str], out_path: str, vocab_size: int = 16000) -> str:
    tok = Tokenizer(models.BPE(unk_token=UNK))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL)
    tok.train(corpus_files, trainer)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(out_path)
    return out_path


class TypoTokenizer:
    def __init__(self, path: str, max_chars: int = 16) -> None:
        self.tok = Tokenizer.from_file(path)
        if self.tok.decoder is None:
            self.tok.decoder = decoders.ByteLevel()
        self.max_chars = max_chars
        self.pad_id = self.tok.token_to_id(PAD)
        self.unk_id = self.tok.token_to_id(UNK)
        self.bos_id = self.tok.token_to_id(BOS)
        self.eos_id = self.tok.token_to_id(EOS)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = self.tok.encode(text).ids
        if add_special:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id}
        return self.tok.decode([i for i in ids if i not in skip])

    def token_strings(self, text: str, add_special: bool = True) -> list[str]:
        pieces = self.tok.encode(text).tokens
        if add_special:
            return [BOS] + pieces + [EOS]
        return pieces

    def char_ids_for_tokens(self, tokens: list[str]) -> list[list[int]]:
        rows = []
        for tok in tokens:
            raw = tok.encode("utf-8", errors="replace")[: self.max_chars]
            row = [b if 0 < b < 128 else 127 for b in raw]
            row += [0] * (self.max_chars - len(row))
            rows.append(row)
        return rows
