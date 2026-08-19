"""
Cross-encoder reranker for Mnemos.

Uses Jina Reranker v2 (multilingual). Scores query↔document pairs for
high-precision reranking after first-stage hybrid retrieval. ~50ms per
batch of 20 documents on CPU.
"""

import os
import threading
import time

from . import constants as _c
from .constants import (FASTEMBED_CACHE, DISABLE_MEM_ARENA,
                        RERANK_BATCH, RERANK_MAX_CHARS, RERANK_MAX_TOKENS)
from . import _resource

_instance = None
_last_used = 0.0
# Resolved once at load: the CML operators THIS reranker's tokenizer cannot
# represent (frozenset; empty = all native, and falsy so the gate reads clean).
_needs_cml_map = frozenset()
_lock = threading.Lock()


def _register_if_custom(TextCrossEncoder):
    """Register RERANKER_MODEL with fastembed when it is not in the catalogue.

    The default reranker (gte-modernbert, since v10.33.0) is not among
    fastembed's built-ins, and neither is whatever a user points
    MNEMOS_RERANKER_MODEL at. add_custom_model refuses names it already
    knows, so this is a no-op for catalogue models and idempotent for
    custom ones.
    """
    try:
        from fastembed.common.model_description import ModelSource
        TextCrossEncoder.add_custom_model(
            model=_c.RERANKER_MODEL,
            sources=ModelSource(hf=_c.RERANKER_MODEL),
            model_file=os.environ.get("MNEMOS_RERANKER_MODEL_FILE",
                                      "onnx/model.onnx"),
            description="registered by mnemos", license="unknown",
            size_in_gb=0.0)
    except ValueError as e:
        if "already registered" not in str(e):
            raise
    except ImportError as e:
        # If fastembed moves ModelSource, silently skipping registration
        # would surface later as an unexplained "model is not supported";
        # name the actual failure instead.
        import sys
        print(f"Mnemos: could not register custom reranker "
              f"{_c.RERANKER_MODEL} with fastembed ({e}); the load will fail "
              "if the model is not in fastembed's catalogue", file=sys.stderr)


def _probe_cml_support(encoder):
    """Return the CML operators this reranker needs spelled out (frozenset).

    The reranker is the component that RESCUES CML: the published ablation has
    single-session-preference R@1 going 53.33% -> 80.00% when it is added, and
    the mechanism is that CML's structural markers are exactly what a
    cross-encoder attends to. A reranker whose vocabulary maps an operator to
    [UNK] cannot do that -- but only for THAT operator, which is why this
    returns the missing set rather than a bool: jina-v2-multilingual carries
    10 of 13 natively and mapping all 13 would throw away the markers the
    ablation showed the cross-encoder using.

    Measured against the real tokenizers (2026-08-19): jina-v2-multilingual
    (Unigram, 250k) loses 3 (because-operator + both FTS5 snippet markers,
    all id=<unk>), jina-v1-turbo-en (BPE, 60k) loses 1, and ms-marco-MiniLM
    (WordPiece, 30k) loses all. Per-model, not a property of "small", so
    probing beats a hardcoded list that goes stale.

    The token id is the ground truth, not the token string: Unigram/XLM-R
    tokenizers echo an unrepresentable character back as its own surface
    piece, so 'sym in tokens' judges it native while the id says <unk>.
    That echo fooled the original bool probe into skipping the map for
    jina-v2-multilingual entirely.

    Detected rather than configured: a flag here is one more thing to set
    wrong, and the tokenizer can answer the question directly.
    """
    from .embed import CML_EMBED_MAP
    missing = set()
    try:
        tok = encoder.model.tokenizer
        unk_id = None
        for cand in ("<unk>", "[UNK]"):
            try:
                unk_id = tok.token_to_id(cand)
            except Exception:
                unk_id = None
            if unk_id is not None:
                break
        for sym in CML_EMBED_MAP:
            if sym.isspace():
                continue
            enc = tok.encode(sym, add_special_tokens=False)
            toks = [t for t in enc.tokens if t not in ("[PAD]", "<pad>")]
            ids = getattr(enc, "ids", None) or []
            if unk_id is not None and unk_id in ids:
                missing.add(sym)
                continue
            if any(t in ("[UNK]", "<unk>") for t in toks):
                missing.add(sym)
                continue
            # Byte-fallback BPE (gte-modernbert and friends) never produces
            # [UNK]: the symbol fragments into UTF-8 byte pieces instead
            # ('\u2713' becomes 'a-circumflex' mojibake). A model represents
            # the operator only if some produced piece still CONTAINS the
            # character; sentencepiece markers are stripped before checking.
            if not any(sym in t.replace("\u2581", "").replace("\u0120", "")
                       for t in toks):
                missing.add(sym)
        return frozenset(missing)
    except Exception:
        return frozenset()


def _pin_tokenizer_shape(encoder):
    """Fix the sequence axis on the tokenizer fastembed already built.

    Deliberately NOT a reimplementation of Jina's pair encoding: the published
    benchmark is a fastembed score stream, and cloning its pre/post-processing
    buys a permanent fidelity job. This only changes the truncation and padding
    policy of the tokenizer object fastembed constructed, so token ids, special
    tokens, pair template and post-processing all stay exactly as they were.

    Reaches through a private attribute, so it degrades to a no-op with a
    warning rather than failing the load if fastembed rearranges internals.
    Returns True when the pin was applied.
    """
    if RERANK_MAX_TOKENS <= 0:
        return False
    try:
        tok = encoder.model.tokenizer
        pad = dict(tok.padding or {})
        tok.enable_truncation(max_length=RERANK_MAX_TOKENS)
        tok.enable_padding(
            length=RERANK_MAX_TOKENS,
            pad_id=pad.get("pad_id", 1),
            pad_token=pad.get("pad_token", "<pad>"),
            pad_type_id=pad.get("pad_type_id", 0),
            direction=pad.get("direction", "right"),
        )
        return True
    except Exception as e:
        import sys
        print(f"Mnemos: could not pin reranker tokenizer shape ({e}); "
              "falling back to fastembed's dynamic padding", file=sys.stderr)
        return False


def _get_reranker():
    global _instance, _last_used, _needs_cml_map
    with _lock:
        if _instance is None:
            _resource.guard_memory()
            try:
                # Heal orphaned download artifacts on THIS path too: the
                # embed-side healer (10.30.1) only runs as a side effect
                # when a fresh process embeds first, and a long-lived server
                # that already holds the embedder never does.
                from .embed import _clean_broken_cache
                _clean_broken_cache()
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                _register_if_custom(TextCrossEncoder)
                kwargs = {
                    "model_name": _c.RERANKER_MODEL,
                    "cache_dir": FASTEMBED_CACHE,
                }
                if DISABLE_MEM_ARENA:
                    kwargs["enable_cpu_mem_arena"] = False
                _instance = TextCrossEncoder(**kwargs)
                _pin_tokenizer_shape(_instance)
                _needs_cml_map = _probe_cml_support(_instance)
                if _needs_cml_map:
                    import sys
                    print(f"Mnemos: {_c.RERANKER_MODEL} cannot represent "
                          f"{len(_needs_cml_map)} CML operator(s) "
                          f"[{' '.join(sorted(_needs_cml_map))}]; spelling "
                          "them out for reranking",
                          file=sys.stderr)
            except ImportError:
                raise ImportError(
                    "Reranker requires fastembed[rerank]. Install with: "
                    "pip install 'fastembed[rerank]'"
                )
        _last_used = time.monotonic()
        return _instance


def maybe_unload(force=False):
    """Drop the reranker if idle longer than MNEMOS_MODEL_IDLE_TTL (or force).

    Mirror of embed.maybe_unload. Opt-in; a no-op at the default TTL of 0. When
    the reranker is unloaded under memory pressure, rerank() catches the
    resulting load failure and returns documents in their pre-rerank (RRF)
    order, so search degrades in precision rather than failing.
    """
    global _instance
    with _lock:
        if _instance is not None and (
            force or (_resource.IDLE_TTL and time.monotonic() - _last_used > _resource.IDLE_TTL)
        ):
            _instance = None
            _resource.trim()
            return True
    return False


def _clip(text):
    """Bound a document's contribution to the sequence axis of the tensor.

    Cross-encoder cost is quadratic in sequence length and the arena retains
    the peak per shape, so the longest memory in a batch sets a permanent
    floor for the whole process. Truncation is on the SCORING copy only; the
    stored memory and everything returned to the caller are untouched.
    CML front-loads the fact, so the head of a memory is the part that decides
    its rank.
    """
    if RERANK_MAX_CHARS <= 0 or len(text) <= RERANK_MAX_CHARS:
        return text
    return text[:RERANK_MAX_CHARS]


def rerank(query: str, documents: list) -> list:
    """Rerank documents by cross-encoder relevance to query.

    documents: list of dicts with at least {"text": ..., "id": ...}
    Returns the same list, sorted by score (highest first), with `_rerank_score` set.
    """
    if not documents:
        return []
    try:
        model = _get_reranker()
        prep = _clip
        if _needs_cml_map:
            from .embed import apply_cml_map
            missing = _needs_cml_map
            prep = lambda t: _clip(apply_cml_map(t, missing))
            # The query gets the same treatment: a cross-encoder scores the
            # PAIR, and a mapped document against an unmapped query would
            # reintroduce the mismatch one level up.
            query = apply_cml_map(query, missing)
        texts = [prep(d.get("text", "")) for d in documents]
        scores = list(model.rerank(query, texts, batch_size=RERANK_BATCH))
    except Exception as e:
        import sys
        print(f"Reranker error: {e}", file=sys.stderr)
        return documents

    for doc, score in zip(documents, scores):
        doc["_rerank_score"] = float(score)
    return sorted(documents, key=lambda d: d.get("_rerank_score", 0), reverse=True)


def rrf_merge(*ranked_lists, k=60):
    """Reciprocal Rank Fusion of multiple ranked ID lists.

    score(id) = sum(1 / (k + rank_in_list_i)) for each list
    """
    scores = {}
    for ids in ranked_lists:
        for rank, mid in enumerate(ids):
            scores[mid] = scores.get(mid, 0) + 1.0 / (k + rank + 1)
    return [mid for mid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
