"""
Resource-awareness helpers for Mnemos: optional idle model unloading and
memory-pressure guarding.

All behaviour here is opt-in via environment variables and defaults to a no-op,
so existing deployments are unaffected. It exists so a lean memory system can
run alongside other services on a small box without permanently pinning the
embedder and reranker in RAM.

    MNEMOS_MODEL_IDLE_TTL   seconds a model may sit idle before the reaper
                            unloads it on its next tick. 0 (default) = never
                            unload, models stay resident as before.
    MNEMOS_MIN_FREE_MB      refuse to load a model when MemAvailable is below
                            this floor, so the search path degrades to a
                            lighter stage instead of risking an OOM kill.
                            0 (default) = off.
    MNEMOS_DISABLE_MEM_ARENA  disable the ONNX Runtime CPU memory arena on
                            every session Mnemos creates (embedder, reranker,
                            NLI scorers), bounding RSS during ACTIVE periods
                            where the idle reaper never gets a window.
                            ~10-15% slower inference. 0 (default) = off.
                            (Lives in constants.py; listed here because this
                            docstring is the resource-controls index.)

Reclaim mechanics (measured): dropping the model reference plus gc frees the
ONNX session and returns its arena to the OS; malloc_trim then hands back the
glibc-side residue. Both steps are needed for RSS to actually fall.
"""

import ctypes
import gc
import os
import threading
import time

IDLE_TTL = int(os.environ.get("MNEMOS_MODEL_IDLE_TTL", "0"))
MIN_FREE_MB = int(os.environ.get("MNEMOS_MIN_FREE_MB", "0"))


def available_mb():
    """Best-effort MemAvailable in MB from /proc/meminfo.

    Returns None when it cannot be read (e.g. non-Linux), which callers treat
    as "no pressure" so the guard never blocks where it cannot measure.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return None
    return None


def guard_memory():
    """Raise MemoryError if free memory is below MNEMOS_MIN_FREE_MB.

    Call this before loading a model. Callers in the search path already catch
    the failure and drop to a lighter stage (vec-only, then FTS5), so this
    turns a potential OOM into graceful degradation. No-op when MIN_FREE_MB=0.
    """
    if MIN_FREE_MB:
        avail = available_mb()
        if avail is not None and avail < MIN_FREE_MB:
            raise MemoryError(f"{avail:.0f} MB available < {MIN_FREE_MB} MB floor")


def trim():
    """Run gc then return freed glibc arena to the OS via malloc_trim.

    gc frees the ONNX session (and its arena); malloc_trim reclaims the
    glibc-held residue gc leaves behind. Safe no-op off glibc.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


_reaper_started = False
_reaper_lock = threading.Lock()


def start_idle_reaper(interval=60, log=None):
    """Start the background idle-model reaper, once per process.

    Dropping an idle ONNX session returns its arena to the OS, which is the
    only mechanism Mnemos has for giving memory back: the ORT CPU arena keeps
    the peak of every tensor shape it has served and never releases it while
    the session lives.

    This used to be started by the stdio entrypoint alone, which is backwards.
    A stdio harness already reclaims everything when its session ends and the
    process exits; the long-lived shared server (v10.27.0) is the deployment
    that has no other reset and it was the one running without a reaper. Both
    transports call this now.

    No thread and no effect at the default TTL of 0.
    """
    global _reaper_started
    if IDLE_TTL <= 0:
        return False
    with _reaper_lock:
        if _reaper_started:
            return False
        _reaper_started = True

    def _reap():
        from . import embed as _embed_mod
        from . import rerank as _rerank_mod
        from . import nli as _nli_mod
        while True:
            time.sleep(interval)
            try:
                _embed_mod.maybe_unload()
                _rerank_mod.maybe_unload()
                # NLI was missing from the tick, so a quiet night still left
                # both DeBERTa sessions and their arenas resident.
                _nli_mod.maybe_unload()
            except Exception:
                pass

    threading.Thread(target=_reap, daemon=True, name="mnemos-model-reaper").start()
    if log:
        log(f"Mnemos: idle model reaper on (TTL {IDLE_TTL}s)")
    return True
