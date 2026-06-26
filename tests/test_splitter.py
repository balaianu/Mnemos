"""Tests for the lossless memory size-guard splitter (mnemos/splitter.py)."""

from mnemos.splitter import (
    split_content, split_is_lossless, needs_split, _nonblank_lines,
)


def _cml_blob(n_facts, prefix="F"):
    """Build a realistic CML-style blob: blank-line-separated blocks of facts."""
    blocks = []
    for b in range(n_facts // 5):
        lines = [f"## Topic {b}"]
        for i in range(5):
            lines.append(f"{prefix}:fact {b}.{i}; some detail about subject {b} item {i} with enough text to add length")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def test_short_content_unchanged():
    c = "F:a single small fact; nothing to split"
    assert split_content(c) == [c]


def test_none_and_empty():
    assert split_content(None) == [""]
    assert split_content("") == [""]


def test_needs_split():
    assert not needs_split("x" * 100, threshold=4000)
    assert needs_split("x" * 5000, threshold=4000)


def test_oversized_splits_into_multiple():
    blob = _cml_blob(200)
    assert len(blob) > 4000
    chunks = split_content(blob, threshold=4000, target=2800)
    assert len(chunks) > 1


def test_chunks_respect_target_or_are_single_line():
    blob = _cml_blob(300)
    chunks = split_content(blob, threshold=4000, target=2800)
    for c in chunks:
        # Each chunk is within target, unless it is a single verbatim line
        # that itself exceeds target (we never break inside a line).
        assert len(c) <= 2800 or len(_nonblank_lines(c)) == 1


def test_lossless_property():
    blob = _cml_blob(300)
    chunks = split_content(blob, threshold=4000, target=2800)
    assert split_is_lossless(blob, chunks)
    # Every fact line preserved exactly, in order.
    assert _nonblank_lines(blob) == [
        ln for c in chunks for ln in _nonblank_lines(c)
    ]


def test_single_giant_line_kept_intact():
    giant = "F:" + ("verbatim detail; " * 400)  # one line, > target, no newlines
    assert len(giant) > 2800
    chunks = split_content(giant, threshold=2000, target=2800)
    # A single un-splittable line must survive verbatim in exactly one chunk.
    assert any(giant in c for c in chunks)
    assert split_is_lossless(giant, chunks)


def test_no_fact_lost_across_many_blocks():
    blob = _cml_blob(1000)  # large, many splits
    chunks = split_content(blob, threshold=4000, target=2800)
    assert len(chunks) >= 5
    assert split_is_lossless(blob, chunks)
    assert sum(len(_nonblank_lines(c)) for c in chunks) == len(_nonblank_lines(blob))


def test_dated_sections_blob():
    blob = "\n\n".join(
        f"## Update: 2026-0{m}-15, topic {m}\nF:detail {m}a; xxxxxxxxxx\nF:detail {m}b; yyyyyyyyyy\nW:watch {m}; zzzzzzzz"
        for m in range(1, 9)
    ) * 8
    chunks = split_content(blob, threshold=4000, target=2800)
    assert split_is_lossless(blob, chunks)
