"""
FastEmbed wrapper for Mnemos.

Uses the FASTEMBED_MODEL ONNX model (default multilingual-e5-large, 1024-dim).
Loads once at startup, ~7ms per embedding on CPU. Embedding families differ in
how they mark documents versus queries; _prefixes() resolves that per model so
callers keep passing a plain "passage" / "query" role.
"""

import hashlib
import threading
import time

from .constants import (
    FASTEMBED_MODEL, FASTEMBED_CACHE, FASTEMBED_DIMS, DISABLE_MEM_ARENA,
)
from . import _resource

_instance = None
_last_used = 0.0
_lock = threading.Lock()


def _get_model():
    global _instance, _last_used
    with _lock:
        if _instance is None:
            _resource.guard_memory()
            from fastembed import TextEmbedding
            kwargs = {
                "model_name": FASTEMBED_MODEL,
                "cache_dir": FASTEMBED_CACHE,
            }
            if DISABLE_MEM_ARENA:
                kwargs["enable_cpu_mem_arena"] = False
            _instance = TextEmbedding(**kwargs)
        _last_used = time.monotonic()
        return _instance


def maybe_unload(force=False):
    """Drop the embedder if it has been idle longer than MNEMOS_MODEL_IDLE_TTL.

    Returns True if a model was unloaded. The next embed() pays a one-off
    reload (about 1-2s on a fast CPU, more on small hardware). Opt-in: with the
    default TTL of 0 this never fires. An in-flight embed() holds its own local
    reference to the model, so unloading here cannot pull the ONNX session out
    from under a query that is already running (CPython refcounting keeps it
    alive until that call returns).
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


def _prefixes():
    """(document, query) prefix pair for the configured embedder.

    Each family has its own convention and applying the wrong one silently
    degrades every vector on both sides of the index. e5 wants "passage: " /
    "query: "; BGE English v1.5 wants a bare passage and an instruction-led
    query. Unknown models get no prefix, because a wrong prefix is worse than
    none.
    """
    m = FASTEMBED_MODEL.lower()
    if "e5" in m:
        return "passage: ", "query: "
    if "bge" in m and "-en" in m:
        return "", "Represent this sentence for searching relevant passages: "
    return "", ""


_dims_checked = False


def _check_dims(actual):
    """Fail loudly, once, when the model and MNEMOS_EMBED_DIMS disagree.

    sqlite-vec already rejects a mismatched insert, but its message names the
    numbers without naming the knob. Switching MNEMOS_EMBED_MODEL and
    forgetting MNEMOS_EMBED_DIMS is the obvious way to get here, so say which
    value to set rather than leaving the caller to infer it.
    """
    global _dims_checked
    if _dims_checked:
        return
    _dims_checked = True
    if actual != FASTEMBED_DIMS:
        raise ValueError(
            f"embedding dimension mismatch: {FASTEMBED_MODEL} produces "
            f"{actual}-dim vectors but MNEMOS_EMBED_DIMS is {FASTEMBED_DIMS}. "
            f"Set MNEMOS_EMBED_DIMS={actual} and re-embed the store "
            f"(drop embed_vec, then `mnemos embed-fill`)."
        )


def embed(texts, prefix="passage"):
    """Embed a list of texts. prefix is 'passage' for docs, 'query' for queries.

    Returns a list of lists of floats (FASTEMBED_DIMS-dim, L2-normalized).
    Returns empty list on failure.
    """
    if not texts:
        return []
    if isinstance(texts, str):
        texts = [texts]
    doc_pfx, qry_pfx = _prefixes()
    pfx = qry_pfx if prefix == "query" else doc_pfx
    prefixed = [f"{pfx}{t}" for t in texts]
    try:
        import math
        model = _get_model()
        # L2-normalize each vector so cosine similarity can be computed as a
        # simple dot product and L2 distance stays bounded in [0, 2]. Recent
        # fastembed versions no longer normalize e5-large output, so we do it
        # here explicitly, all downstream thresholds (dedup, contradiction
        # detection) assume unit-norm vectors.
        out = []
        for vec in model.embed(prefixed):
            v = list(vec)
            norm = math.sqrt(sum(x * x for x in v))
            if norm > 0:
                v = [x / norm for x in v]
            out.append(v)
    except Exception as e:
        import sys
        print(f"FastEmbed error: {e}", file=sys.stderr)
        return []
    if out:
        _check_dims(len(out[0]))
    return out


def text_hash(text: str) -> str:
    """SHA256 hash of text, used to detect changes for re-embedding."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Nyx consolidation rewrites these bookkeeping tags every cycle (merge/split
# markers). They carry no retrieval signal, so folding them into the embed-text
# only churns the vector and, worse, invalidates the coherence hash on every
# store that has ever been consolidated. Excluded from the embed-text since
# v10.22.0 so the canonical text is stable across consolidation.
_NYX_TAG_EXACT = frozenset({"consolidated", "nyx-split", "nyx-cycle",
                            "synthesized", "bridge"})
_NYX_TAG_PREFIX = ("merged-into", "split-from", "split-part")


def stable_tags(tags: str) -> str:
    """Drop Nyx-internal bookkeeping tags, keep retrieval-relevant ones."""
    if not tags:
        return ""
    kept = []
    for t in tags.split(","):
        s = t.strip()
        if not s:
            continue
        low = s.lower()
        if low in _NYX_TAG_EXACT or low.startswith(_NYX_TAG_PREFIX):
            continue
        kept.append(s)
    return ", ".join(kept)


def prep_memory_text(project, content, tags="", mem_type="", layer=""):
    """Build the canonical text representation used for embedding a memory.

    Combines project, type, layer, content, and retrieval-relevant tags so the
    embedding captures the metadata that affects retrieval. Nyx bookkeeping tags
    (merge/split markers) are excluded: they churn every consolidation cycle and
    carry no retrieval signal, so including them only destabilizes the vector
    and the coherence hash.
    """
    parts = [project]
    if mem_type and mem_type != "fact":
        parts.append(f"[{mem_type}]")
    if layer and layer != "semantic":
        parts.append(f"[{layer}]")
    parts.append(content)
    stable = stable_tags(tags)
    if stable:
        parts.append(stable)
    return " ".join(parts).strip()
