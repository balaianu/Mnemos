"""Language-coverage heuristics for the model-tier warnings.

An English-only default fails SILENTLY on a non-English store: nothing errors,
retrieval is just quietly mediocre, and nothing attributes it. That is the
same failure shape as a dimension mismatch or a provenance split, so it gets
the same treatment: detect and say so.

Heuristic, not language ID. A non-ASCII-letter ratio catches accented
European text, CJK, Cyrillic, Thai and Arabic without any model. It cannot
distinguish Swedish from German and does not need to; the question is only
"is there meaningful non-English content in front of an English-only model".
CML operator symbols and emoji are excluded so notation does not count as
language.
"""

# Encoder/reranker ids that only understand English. Family substrings, not
# exact names, so obvious variants match. Multilingual models never appear:
# unknown ids are treated as capable, because a false "your setup is fine"
# is worse than a false warning the user can dismiss.
_ENGLISH_ONLY = ("-en-", "-en_", "bge-small-en", "bge-base-en", "bge-large-en",
                 "modernbert", "ms-marco", "minilm-l")


def _ends_with_en(m):
    return m.endswith("-en") or m.endswith("_en")

def is_english_only(model_id):
    m = (model_id or "").lower()
    if "multilingual" in m or "-e5-" in m or m.endswith("e5-large"):
        return False
    return _ends_with_en(m) or any(t in m for t in _ENGLISH_ONLY)


def non_english_share(text):
    """Share of letters outside ASCII, ignoring symbols/emoji/operators."""
    letters = 0
    non_ascii = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        if ord(ch) > 127:
            non_ascii += 1
    return (non_ascii / letters) if letters else 0.0


def scan_contents(rows, per_doc_threshold=0.005):
    """rows: iterable of content strings. Returns (affected, total).

    A memory counts as non-English when more than per_doc_threshold of its
    letters are non-ASCII. 0.5% is deliberately very low, calibrated against
    a real English-primary store whose Swedish TERMS OF ART are its retrieval
    keys: those memories sit at 0.5-2% non-ASCII letters, and they are
    exactly the case where a multilingual reranker still earns its place.
    Pure-English stores measure 0 and never trip it.
    """
    affected = total = 0
    for c in rows:
        total += 1
        if non_english_share(c or "") > per_doc_threshold:
            affected += 1
    return affected, total
