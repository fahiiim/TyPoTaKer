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


def _split_tokens(text: str) -> tuple[list[str], list[str]]:
    """Split text while retaining every separator exactly."""
    matches = list(re.finditer(r"\w+|[^\w\s]", text))
    if not matches:
        return [], [text]
    tokens = [m.group(0) for m in matches]
    gaps = [text[: matches[0].start()]]
    gaps.extend(text[a.end() : b.start()] for a, b in zip(matches, matches[1:]))
    gaps.append(text[matches[-1].end() :])
    return tokens, gaps


def _merge_deleted_gaps(left: str, right: str) -> str:
    if left.isspace() and right.isspace():
        if "\n" in left or "\n" in right:
            return "\n\n" if "\n\n" in left + right else "\n"
        return " " if left or right else ""
    return left + right


def _delete_token(tokens: list[str], gaps: list[str], index: int) -> None:
    tokens.pop(index)
    gaps[index] = _merge_deleted_gaps(gaps[index], gaps[index + 1])
    gaps.pop(index + 1)


def _render_tokens(tokens: list[str], gaps: list[str]) -> str:
    return "".join(gap + token for gap, token in zip(gaps, tokens)) + gaps[-1]


def corrupt_sentence(text: str, rng: random.Random | None = None, p: float = 0.7) -> str:
    """Return a noisy version. ~30% of sentences stay clean (identity KEEP training)."""
    rng = rng or random.Random()
    if rng.random() > p:
        return text
    words, gaps = _split_tokens(text)
    if not words:
        return text
    n_ops = rng.randint(1, 3)
    for _ in range(n_ops):
        op = rng.choice(
            ["typo", "confusion", "drop_article", "prep", "case", "sv_s"]
        )
        if op == "typo":
            candidates = [i for i, word in enumerate(words) if word.isalpha() and len(word) >= 3]
            if not candidates:
                continue
            idx = rng.choice(candidates)
            words[idx] = _typo_word(words[idx], rng)
        elif op == "confusion":
            src, dst = rng.choice(CONFUSIONS)
            candidates = [i for i, word in enumerate(words) if word.lower() == src.lower()]
            if candidates:
                words[rng.choice(candidates)] = dst
        elif op == "drop_article":
            arts = [i for i, w in enumerate(words) if w.lower() in ARTICLES]
            if arts:
                _delete_token(words, gaps, rng.choice(arts))
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
    return _render_tokens(words, gaps)
