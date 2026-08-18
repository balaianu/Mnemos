"""
FastEmbed wrapper for Mnemos.

Uses the FASTEMBED_MODEL ONNX model (default multilingual-e5-large, 1024-dim).
Loads once at startup, ~7ms per embedding on CPU. Embedding families differ in
how they mark documents versus queries; _prefixes() resolves that per model so
callers keep passing a plain "passage" / "query" role.
"""

import hashlib
import re
import sys
import threading
import time
from pathlib import Path

from .constants import (
    FASTEMBED_MODEL, FASTEMBED_CACHE, FASTEMBED_DIMS, DISABLE_MEM_ARENA,
    EMBED_NORMALIZE_CML, EMBED_MODEL_EXPLICIT, EMBED_DIMS_EXPLICIT,
    EMBED_NORMALIZE_EXPLICIT, EMBED_BATCH,
)
from . import _resource

_WS_RE = re.compile(r"  +")

_instance = None
_last_used = 0.0
_lock = threading.Lock()

# Files younger than this are assumed to belong to an active download and
# are left alone. 1 hour is conservative: a 2.2G model download finishes in
# minutes on any reasonable connection, and concurrent mnemos processes
# (MCP server + CLI + consolidation cron) can overlap safely.
_INCOMPLETE_STALE_SECONDS = 3600


def _clean_broken_cache():
    """Remove orphaned download artifacts that prevent model loading.

    huggingface_hub leaves ``.incomplete`` temp files when a download is
    interrupted and never garbage-collects them. A failed initial download
    can also leave ``refs/main`` empty (0 bytes), which causes
    ``snapshot_download(local_files_only=True)`` to return the ``snapshots/``
    parent dir instead of the hash subdir, and onnxruntime then fails with
    NoSuchFile because ``snapshots/model.onnx`` doesn't exist. Both
    conditions block the model from ever loading and compound with each
    retry: every failed attempt adds another dead partial, filling the disk
    and making the next download fail even sooner.

    Safe with concurrent downloads: only ``.incomplete`` files older than
    ``_INCOMPLETE_STALE_SECONDS`` are removed (an active download keeps its
    file's mtime fresh). For empty ``refs/main``, only the file itself is
    deleted, not the model dir, so already-downloaded small files survive
    and ``snapshot_download`` raises ``LocalEntryNotFoundError`` (which
    fastembed catches and falls through to a network re-download).
    """
    cache = Path(FASTEMBED_CACHE)
    if not cache.is_dir():
        return

    now = time.time()
    for f in cache.glob("models--*/blobs/*.incomplete"):
        try:
            if now - f.stat().st_mtime > _INCOMPLETE_STALE_SECONDS:
                f.unlink()
        except OSError:
            pass

    for refs_main in cache.glob("models--*/refs/main"):
        try:
            if refs_main.stat().st_size == 0:
                refs_main.unlink()
                print(
                    f"mnemos: removed empty refs/main for "
                    f"{refs_main.parent.parent.name}",
                    file=sys.stderr,
                )
        except OSError:
            pass


def _get_model():
    global _instance, _last_used
    with _lock:
        if _instance is None:
            _resource.guard_memory()
            _clean_broken_cache()
            from fastembed import TextEmbedding
            kwargs = {
                "model_name": effective_model(),
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


# Only the operators an English WordPiece vocabulary cannot represent. The
# survivors (-> <-> <- @ ~ null >) are left alone: substituting a token the
# model already has would change the text for no measured reason.
CML_EMBED_MAP = {
    "\u2235": " because ",      # therefore-because
    "\u2234": " therefore ",
    "\u25b3": " changed from ",
    "\u26a0": " warning ",
    "\u2713": " confirmed ",
    "\u2717": " rejected ",
    "\u27ea": " ",              # FTS5 snippet markers, not semantic
    "\u27eb": " ",
    "\u2550": " ",              # box-drawing separators in prose memories
}


def apply_cml_map(text):
    """Substitute CML operators for the words they stand for. Ungated.

    Split out from normalize_cml so the reranker can decide for itself: the
    embedder applies this on a config flag, while the reranker applies it only
    when its own tokenizer cannot represent the operators.
    """
    if not text:
        return text
    for sym, word in CML_EMBED_MAP.items():
        if sym in text:
            text = text.replace(sym, word)
    return _WS_RE.sub(" ", text)


def normalize_cml(text):
    """apply_cml_map, gated on MNEMOS_EMBED_NORMALIZE_CML.

    Applied to every text on its way to the encoder, documents and queries
    alike, so both sides of the index are built from the same distribution.
    """
    if not effective_normalize():
        return text
    return apply_cml_map(text)


def embed_model_id():
    """Provenance string for embed_meta.model.

    Normalization changes every vector without changing the model name, so it
    has to be part of the identity or doctor's provenance check cannot see a
    flip and the store silently mixes two geometries.
    """
    return effective_model() + ("+cmlnorm" if effective_normalize() else "")


# Effective embedder config. The env vars seed it; a populated store can pin it
# to whatever it was actually built with (see adopt_store_config). A store's
# vectors are only comparable to vectors from the same model, dimensions and
# normalization setting, so the store's own history outranks a default that
# happens to be current. An explicitly exported env var still wins: that is a
# deliberate instruction, and the operator is told to re-embed.
_effective = {
    "model": FASTEMBED_MODEL,
    "dims": FASTEMBED_DIMS,
    "normalize": EMBED_NORMALIZE_CML,
}


def effective_model():
    return _effective["model"]


def effective_dims():
    return _effective["dims"]


def effective_normalize():
    return _effective["normalize"]


def adopt_store_config(model_id, dims=None):
    """Pin the embedder to what a populated store was built with.

    `model_id` is an embed_meta.model value, so it may carry the "+cmlnorm"
    suffix. `dims` comes from the declared width of the vector index, which is
    authoritative and needs no model catalogue lookup.

    Fields the operator set explicitly are left alone. Returns a dict of what
    actually changed, empty when nothing did, so callers can surface it rather
    than silently reconfiguring under the user.
    """
    global _instance, _dims_checked
    if not model_id:
        return {}
    normalize = model_id.endswith("+cmlnorm")
    model = model_id[: -len("+cmlnorm")] if normalize else model_id

    changed = {}
    if not EMBED_MODEL_EXPLICIT and model != _effective["model"]:
        changed["model"] = (_effective["model"], model)
        _effective["model"] = model
    if not EMBED_NORMALIZE_EXPLICIT and normalize != _effective["normalize"]:
        changed["normalize"] = (_effective["normalize"], normalize)
        _effective["normalize"] = normalize
    if dims and not EMBED_DIMS_EXPLICIT and dims != _effective["dims"]:
        changed["dims"] = (_effective["dims"], dims)
        _effective["dims"] = dims

    if changed:
        # The loaded session belongs to the old config; drop it so the next
        # embed() builds under the adopted one.
        with _lock:
            _instance = None
        _dims_checked = False
    return changed


def _prefixes():
    """(document, query) prefix pair for the configured embedder.

    Each family has its own convention and applying the wrong one silently
    degrades every vector on both sides of the index. e5 wants "passage: " /
    "query: "; BGE English v1.5 wants a bare passage and an instruction-led
    query. Unknown models get no prefix, because a wrong prefix is worse than
    none.
    """
    m = effective_model().lower()
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
    if actual != effective_dims():
        raise ValueError(
            f"embedding dimension mismatch: {effective_model()} produces "
            f"{actual}-dim vectors but the configured width is "
            f"{effective_dims()}. "
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
    prefixed = [f"{pfx}{normalize_cml(t)}" for t in texts]
    try:
        import math
        model = _get_model()
        # L2-normalize each vector so cosine similarity can be computed as a
        # simple dot product and L2 distance stays bounded in [0, 2]. Recent
        # fastembed versions no longer normalize e5-large output, so we do it
        # here explicitly, all downstream thresholds (dedup, contradiction
        # detection) assume unit-norm vectors.
        out = []
        for vec in model.embed(prefixed, batch_size=EMBED_BATCH):
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
