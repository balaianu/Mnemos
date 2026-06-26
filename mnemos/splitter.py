"""
Memory size-guard splitter for Mnemos.

Keeps individual memories atomic by splitting oversized content into smaller
chunks, losslessly: every non-blank original line lands in exactly one chunk,
in original order, nothing paraphrased or dropped. CML memories (one fact per
line, blank-line-separated blocks) split cleanly along block and line
boundaries.

Design note (load-bearing principle: a memory system should store and retrieve,
not think endlessly): the bulk splitter is pure mechanical text work with no
LLM and no DB access. LLM-assisted topical clustering for giant or prose
blobs is a separate Phase 1 concern, layered on top, used only on the handful
of very large memories. This module stays dependency-free (stdlib only) so the
store hot path can import it without pulling in consolidation or numpy.
"""

import os

SPLIT_THRESHOLD = int(os.environ.get("MNEMOS_SPLIT_THRESHOLD", "4000"))
SPLIT_TARGET = int(os.environ.get("MNEMOS_SPLIT_TARGET", "2800"))


def split_enabled():
    """Whether auto-splitting is on. Default on; set MNEMOS_SPLIT_ENABLED=0 to disable."""
    return os.environ.get("MNEMOS_SPLIT_ENABLED", "1").lower() not in ("0", "false", "no", "")


def _blocks(content):
    """Group lines into blocks at blank-line boundaries.

    Each block is a list of consecutive non-blank lines (a paragraph or a CML
    block such as a heading plus the facts under it). Blank lines are treated
    purely as separators and are not preserved as content.
    """
    blocks = []
    cur = []
    for line in content.split("\n"):
        if line.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def split_content(content, threshold=None, target=None):
    """Split content into atomic chunks, losslessly.

    Returns a list of chunk strings. If content is within `threshold` (or empty
    or None), returns a single-element list with the content unchanged.
    Otherwise it packs whole blocks (then, for an oversized single block, whole
    lines) into chunks of at most `target` characters. It never breaks inside a
    line, so every fact stays verbatim and every non-blank line appears in
    exactly one chunk, in original order. Blank separator lines are normalized.

    Verify the guarantee with split_is_lossless(original, chunks).
    """
    threshold = SPLIT_THRESHOLD if threshold is None else threshold
    target = SPLIT_TARGET if target is None else target
    if not content or len(content) <= threshold:
        return [content if content is not None else ""]

    chunks = []
    cur = []          # list of lines accumulated for the current chunk
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            # Drop only leading/trailing blank separator lines; content lines
            # are kept verbatim so the lossless invariant holds byte for byte.
            lines = cur
            while lines and lines[0].strip() == "":
                lines.pop(0)
            while lines and lines[-1].strip() == "":
                lines.pop()
            if lines:
                chunks.append("\n".join(lines))
            cur = []
            cur_len = 0

    for block in _blocks(content):
        block_text = "\n".join(block)
        block_len = len(block_text) + 2  # account for the blank-line separator

        if block_len > target:
            # Oversized single block: flush what we have, then pack its lines.
            flush()
            for line in block:
                add = len(line) + 1
                if cur and cur_len + add > target:
                    flush()
                cur.append(line)
                cur_len += add
            flush()
            continue

        if cur and cur_len + block_len > target:
            flush()
        if cur:
            cur.append("")  # blank line between blocks within a chunk
            cur_len += 1
        cur.extend(block)
        cur_len += block_len

    flush()
    return chunks or [content]


def _nonblank_lines(text):
    return [ln for ln in (text or "").split("\n") if ln.strip()]


def split_is_lossless(original, chunks):
    """True iff every non-blank line of `original` appears across `chunks`
    exactly once, in the original order. This is the safety invariant: the
    splitter must never drop, duplicate, reorder, or paraphrase a fact.
    """
    orig = _nonblank_lines(original)
    rebuilt = []
    for c in chunks:
        rebuilt.extend(_nonblank_lines(c))
    return orig == rebuilt


def needs_split(content, threshold=None):
    threshold = SPLIT_THRESHOLD if threshold is None else threshold
    return bool(content) and len(content) > threshold
