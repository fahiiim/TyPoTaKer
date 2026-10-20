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

# Correct -> intentionally corrupted. These cover frequent keyboard/spelling
# mistakes that are useful for a desktop typing assistant.
COMMON_TYPOS = [
    ("the", "teh"),
    ("receive", "recieve"),
    ("separate", "seperate"),
    ("address", "adress"),
    ("definitely", "definately"),
    ("occurred", "occured"),
    ("environment", "enviroment"),
    ("weird", "wierd"),
    ("because", "becuase"),
    ("tomorrow", "tomorow"),
    ("their", "thier"),
    ("until", "untill"),
    ("coming", "comming"),
    ("successful", "succesful"),
    ("necessary", "neccessary"),
    ("calendar", "calender"),
    ("available", "avaliable"),
    ("immediately", "imediately"),
    ("decision", "descision"),
    ("knowledge", "knowlege"),
]

# Clean token sequences -> common grammar errors. We only apply transformations
# when the full local pattern matches, avoiding the old arbitrary "add s" rule.
GRAMMAR_PATTERNS = [
    (("i", "have"), ("i", "has")),
    (("she", "has"), ("she", "have")),
    (("he", "has"), ("he", "have")),
    (("they", "are"), ("they", "is")),
    (("we", "are"), ("we", "is")),
    (("we", "were"), ("we", "was")),
    (("they", "were"), ("they", "was")),
    (("she", "goes"), ("she", "go")),
    (("he", "goes"), ("he", "go")),
    (("it", "works"), ("it", "work")),
    (("the", "server", "listens"), ("the", "server", "listen")),
    (("everyone", "is"), ("everyone", "are")),
    (("one", "of", "the"), ("one", "of", "a")),
]


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


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


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


def _apply_common_typo(words: list[str], rng: random.Random) -> bool:
    options: list[tuple[int, str]] = []
    for correct, corrupted in COMMON_TYPOS:
        options.extend((i, corrupted) for i, word in enumerate(words) if word.lower() == correct)
    if not options:
        return False
    index, corrupted = rng.choice(options)
    words[index] = _match_case(words[index], corrupted)
    return True


def _apply_grammar_pattern(words: list[str], rng: random.Random) -> bool:
    options: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []
    lowered = [word.lower() for word in words]
    for clean, noisy in GRAMMAR_PATTERNS:
        width = len(clean)
        for index in range(len(words) - width + 1):
            if tuple(lowered[index : index + width]) == clean:
                options.append((index, clean, noisy))
    if not options:
        return False
    index, clean, noisy = rng.choice(options)
    for offset, replacement in enumerate(noisy):
        words[index + offset] = _match_case(words[index + offset], replacement)
    return True


def _remove_apostrophe(words: list[str], gaps: list[str], rng: random.Random) -> bool:
    candidates = [i for i, word in enumerate(words) if word == "'"]
    if not candidates:
        return False
    _delete_token(words, gaps, rng.choice(candidates))
    return True


def corrupt_sentence(
    text: str,
    rng: random.Random | None = None,
    p: float = 0.7,
    min_ops: int = 1,
    max_ops: int = 3,
) -> str:
    """Return realistic synthetic noise while retaining all original separators.

    ``p`` is the probability of corrupting a unit. ``min_ops`` and ``max_ops``
    let training recipes control the density of meaningful edit supervision.
    """
    rng = rng or random.Random()
    if min_ops < 1 or max_ops < min_ops:
        raise ValueError("expected 1 <= min_ops <= max_ops")
    if rng.random() > p:
        return text
    words, gaps = _split_tokens(text)
    if not words:
        return text
    n_ops = rng.randint(min_ops, max_ops)
    applied = 0
    attempts = n_ops * 4
    for _ in range(attempts):
        if applied >= n_ops:
            break
        op = rng.choice(
            [
                "common_typo",
                "common_typo",
                "typo",
                "confusion",
                "drop_article",
                "prep",
                "case",
                "grammar",
                "apostrophe",
            ]
        )
        changed = False
        if op == "typo":
            candidates = [i for i, word in enumerate(words) if word.isalpha() and len(word) >= 3]
            if not candidates:
                continue
            idx = rng.choice(candidates)
            replacement = _typo_word(words[idx], rng)
            changed = replacement != words[idx]
            words[idx] = replacement
        elif op == "common_typo":
            changed = _apply_common_typo(words, rng)
        elif op == "confusion":
            src, dst = rng.choice(CONFUSIONS)
            candidates = [i for i, word in enumerate(words) if word.lower() == src.lower()]
            if candidates:
                index = rng.choice(candidates)
                words[index] = _match_case(words[index], dst)
                changed = True
        elif op == "drop_article":
            arts = [i for i, w in enumerate(words) if w.lower() in ARTICLES]
            if arts:
                _delete_token(words, gaps, rng.choice(arts))
                changed = True
        elif op == "prep":
            idxs = [i for i, w in enumerate(words) if w.lower() in PREP]
            if idxs:
                index = rng.choice(idxs)
                choices = [prep for prep in PREP if prep != words[index].lower()]
                words[index] = _match_case(words[index], rng.choice(choices))
                changed = True
        elif op == "case":
            candidates = [i for i, word in enumerate(words) if word.isalpha() and word != word.lower()]
            if candidates:
                index = rng.choice(candidates)
                words[index] = words[index].lower()
                changed = True
        elif op == "grammar":
            changed = _apply_grammar_pattern(words, rng)
        elif op == "apostrophe":
            changed = _remove_apostrophe(words, gaps, rng)
        if changed:
            applied += 1
    return _render_tokens(words, gaps)
