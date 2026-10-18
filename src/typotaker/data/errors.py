"""Synthetic grammar/typo noise. No external GEC corpus required."""

from __future__ import annotations

import random
import re

CONFUSIONS = [
    ("their", "there"),
    ("there", "their"),
    ("they're", "their"),
    ("its", "it's"),
    ("it's", "its"),
    ("your", "you're"),
    ("you're", "your"),
    ("then", "than"),
    ("than", "then"),
    ("affect", "effect"),
    ("effect", "affect"),
    ("to", "too"),
    ("too", "to"),
    ("lose", "loose"),
    ("a", "an"),
    ("an", "a"),
]
PREP = ["in", "on", "at", "to", "for", "from", "with", "by"]
ARTICLES = {"a", "an", "the"}
KB_NEIGHBORS = {
    "a": "sqwsz",
    "e": "wsdr",
    "i": "ujko",
    "o": "iklp",
    "s": "awedxz",
    "t": "rfgy",
    "n": "bhjm",
}


def _typo_word(word: str, rng: random.Random) -> str:
    if len(word) < 3 or not word.isalpha():
        return word
    op = rng.choice(["swap", "drop", "dup", "neighbor"])
    i = rng.randrange(len(word) - 1)
    chars = list(word)
    if op == "swap":
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif op == "drop":
        del chars[i]
    elif op == "dup":
        chars.insert(i, chars[i])
    else:
        ch = chars[i].lower()
        nbrs = KB_NEIGHBORS.get(ch, "aeiou")
        chars[i] = rng.choice(nbrs)
        if word[i].isupper():
            chars[i] = chars[i].upper()
    return "".join(chars)


def corrupt_sentence(text: str, rng: random.Random | None = None, p: float = 0.7) -> str:
    """Return a noisy version. ~30% of sentences stay clean (identity KEEP training)."""
    rng = rng or random.Random()
    if rng.random() > p:
        return text
    words = re.findall(r"\w+|[^\w\s]", text)
    if not words:
        return text
    n_ops = rng.randint(1, 3)
    for _ in range(n_ops):
        op = rng.choice(
            ["typo", "confusion", "drop_article", "prep", "case", "sv_s"]
        )
        if op == "typo":
            idx = rng.randrange(len(words))
            words[idx] = _typo_word(words[idx], rng)
        elif op == "confusion":
            joined = " ".join(words)
            src, dst = rng.choice(CONFUSIONS)
            if re.search(rf"\b{re.escape(src)}\b", joined, flags=re.I):
                joined = re.sub(rf"\b{re.escape(src)}\b", dst, joined, count=1, flags=re.I)
                words = re.findall(r"\w+|[^\w\s]", joined)
        elif op == "drop_article":
            arts = [i for i, w in enumerate(words) if w.lower() in ARTICLES]
            if arts:
                del words[rng.choice(arts)]
        elif op == "prep":
            idxs = [i for i, w in enumerate(words) if w.lower() in PREP]
            if idxs:
                words[rng.choice(idxs)] = rng.choice(PREP)
        elif op == "case":
            idx = rng.randrange(len(words))
            if words[idx].isalpha():
                words[idx] = words[idx].lower() if rng.random() < 0.5 else words[idx].upper()
        elif op == "sv_s":
            # crude 3sg: drop or add trailing s on a verb-like token
            cands = [i for i, w in enumerate(words) if w.isalpha() and 3 <= len(w) <= 10]
            if cands:
                i = rng.choice(cands)
                words[i] = words[i][:-1] if words[i].endswith("s") else words[i] + "s"
    out = ""
    for i, w in enumerate(words):
        if i > 0 and re.match(r"\w", w) and re.match(r"\w", words[i - 1]):
            out += " "
        out += w
    return out
