"""Align noisy vs clean token ids into compact edit labels.

Label layout (matches ModelConfig.n_edit_labels):
  0 KEEP
  1 DELETE
  2 CASE_CAP
  3 CASE_LOWER
  4 .. 4+R-1        REPLACE with frequent token class
  4+R .. 4+R+A-1    APPEND frequent token class
Unknown replacements fall back to DELETE (the decoder/app can re-type).
"""

from __future__ import annotations

from difflib import SequenceMatcher

KEEP, DELETE, CASE_CAP, CASE_LOWER = 0, 1, 2, 3
N_BASIC = 4


def build_freq_maps(token_ids: list[int], k_replace: int, k_append: int) -> tuple[dict[int, int], dict[int, int]]:
    from collections import Counter

    counts = Counter(token_ids)
    top_r = [tid for tid, _ in counts.most_common(k_replace)]
    top_a = [tid for tid, _ in counts.most_common(k_append)]
    replace_map = {tid: N_BASIC + i for i, tid in enumerate(top_r)}
    append_map = {tid: N_BASIC + k_replace + i for i, tid in enumerate(top_a)}
    return replace_map, append_map


def align_edits(
    src_ids: list[int],
    tgt_ids: list[int],
    replace_map: dict[int, int],
    append_map: dict[int, int],
) -> list[int]:
    """One label per source token."""
    sm = SequenceMatcher(a=src_ids, b=tgt_ids, autojunk=False)
    labels = [KEEP] * len(src_ids)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            for i in range(i1, i2):
                labels[i] = DELETE
        elif tag == "replace":
            src_span = src_ids[i1:i2]
            tgt_span = tgt_ids[j1:j2]
            for k, sid in enumerate(src_span):
                if k < len(tgt_span) and tgt_span[k] in replace_map:
                    labels[i1 + k] = replace_map[tgt_span[k]]
                else:
                    labels[i1 + k] = DELETE
            extra = tgt_span[len(src_span) :]
            if extra and i2 > 0 and extra[0] in append_map:
                labels[i2 - 1] = append_map[extra[0]]
        elif tag == "insert":
            if i1 == 0:
                continue
            if tgt_ids[j1] in append_map:
                labels[i1 - 1] = append_map[tgt_ids[j1]]
    return labels
