#!/usr/bin/env python3
"""
Can jina-embeddings-v3 be loaded and used at all under a sane memory ceiling?

The 2026-08-19 attempt died as "UNBENCHABLE via fastembed (24GB cgroup
thrash, killed)", so the embedder was never actually measured against
e5-large -- the multilingual tier kept e5 by default, not by evidence.
This probe answers the narrow question before any bench is launched:
does it load, what does it cost, and does it emit sane 1024-dim vectors.

Usage: MNEMOS_EMBED_MODEL=... probe_load.py   (run under a MemoryMax cap)
"""
import os, resource, sys, time

MODEL = os.environ.get("PROBE_MODEL", "jinaai/jina-embeddings-v3")
TEXTS = [
    "F:epsilon runs ubuntu 24.04 with 64gb ecc ram",
    "BRF Ängssätra styrelsemöte kallelse skickas 14 dagar innan",
    "D:the version gate is a range, not an allowlist",
]

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

t0 = time.time()
print(f"model={MODEL}", flush=True)
print(f"  rss after import          {rss_gb():6.2f} GB", flush=True)

from fastembed import TextEmbedding
print(f"  rss after fastembed impt  {rss_gb():6.2f} GB  (+{time.time()-t0:.1f}s)", flush=True)

kwargs = {"model_name": MODEL, "cache_dir": "/root/.cache/fastembed"}
if os.environ.get("PROBE_DISABLE_ARENA") == "1":
    kwargs["providers"] = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]
    print("  (arena_extend_strategy=kSameAsRequested)", flush=True)

t1 = time.time()
model = TextEmbedding(**kwargs)
print(f"  rss after model load      {rss_gb():6.2f} GB  (+{time.time()-t1:.1f}s)", flush=True)

t2 = time.time()
vecs = [list(v) for v in model.embed(TEXTS)]
print(f"  rss after {len(TEXTS)} embeds     {rss_gb():6.2f} GB  (+{time.time()-t2:.1f}s)", flush=True)
print(f"  dims={len(vecs[0])}  norm0={sum(x*x for x in vecs[0])**0.5:.4f}", flush=True)

t3 = time.time()
batch = [f"filler passage number {i} about servers and memory systems" for i in range(64)]
_ = list(model.embed(batch))
print(f"  rss after 64-batch        {rss_gb():6.2f} GB  (+{time.time()-t3:.1f}s)", flush=True)
print(f"PEAK {rss_gb():.2f} GB  total {time.time()-t0:.1f}s", flush=True)
