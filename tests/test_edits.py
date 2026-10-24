from __future__ import annotations

import pytest

from typotaker.data.edits import (
    align_edits,
    build_freq_maps,
    CASE_CAP,
    CASE_LOWER,
    DELETE,
    KEEP,
    N_BASIC,
)


def test_build_freq_maps():
    # Frequency counts: 10 appears 4 times, 20 appears 3 times, 30 appears 2 times, 40 appears 1 time
    tokens = [10, 10, 10, 10, 20, 20, 20, 30, 30, 40]
    rep_map, app_map = build_freq_maps(tokens, k_replace=2, k_append=3)

    assert len(rep_map) == 2
    assert 10 in rep_map and 20 in rep_map
    assert rep_map[10] == N_BASIC  # 4
    assert rep_map[20] == N_BASIC + 1  # 5

    assert len(app_map) == 3
    assert 10 in app_map and 20 in app_map and 30 in app_map
    assert app_map[10] == N_BASIC + 2  # 6
    assert app_map[20] == N_BASIC + 2 + 1  # 7
    assert app_map[30] == N_BASIC + 2 + 2  # 8


def test_align_edits_keep_identical():
    src = [10, 20, 30]
    tgt = [10, 20, 30]
    rep_map = {10: 4, 20: 5}
    app_map = {30: 6}
    labels = align_edits(src, tgt, rep_map, app_map)
    assert labels == [KEEP, KEEP, KEEP]


def test_align_edits_delete():
    src = [10, 20, 30]
    tgt = [10, 30]  # 20 is deleted
    rep_map = {}
    app_map = {}
    labels = align_edits(src, tgt, rep_map, app_map)
    assert labels == [KEEP, DELETE, KEEP]


def test_align_edits_replace_known_and_unknown():
    src = [10, 99, 30]
    tgt = [10, 25, 30]  # 99 -> 25
    
    # 25 is in replace_map
    rep_map = {25: 4}
    app_map = {}
    labels = align_edits(src, tgt, rep_map, app_map)
    assert labels == [KEEP, 4, KEEP]

    # 25 is NOT in replace_map (falls back to DELETE)
    labels_fallback = align_edits(src, tgt, {}, app_map)
    assert labels_fallback == [KEEP, DELETE, KEEP]


def test_align_edits_append():
    src = [10, 30]
    tgt = [10, 20, 30]  # 20 inserted after 10
    app_map = {20: 7}
    labels = align_edits(src, tgt, {}, app_map)
    assert labels == [7, KEEP]  # 7 is appended to token 10


def test_align_edits_casing():
    src_ids = [10, 20]
    tgt_ids = [11, 20]
    src_pieces = ["hello", "world"]
    tgt_pieces = ["Hello", "world"]

    labels = align_edits(
        src_ids,
        tgt_ids,
        replace_map={},
        append_map={},
        src_pieces=src_pieces,
        tgt_pieces=tgt_pieces,
    )
    assert labels == [CASE_CAP, KEEP]

    # Lowercase test
    src_pieces_cap = ["Hello", "world"]
    tgt_pieces_low = ["hello", "world"]
    labels_low = align_edits(
        tgt_ids,
        src_ids,
        replace_map={},
        append_map={},
        src_pieces=src_pieces_cap,
        tgt_pieces=tgt_pieces_low,
    )
    assert labels_low == [CASE_LOWER, KEEP]
