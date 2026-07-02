#!/usr/bin/env python3
"""NLI-vs-Jina bench for the Mnemos store decision layer (dedup + contradiction).

Arms:
  native  - pair texts as stored (mixed sv/en)
  english - pair texts with cached Haiku translations substituted (translate.py)

Scorers:
  vec      - e5-large cosine (current dedup blocker signal)
  jina     - Jina v2 reranker sigmoid, max of both directions (current M7 signal)
  4 NLI models - P(entailment)/P(contradiction) per direction via transformers

Tasks:
  contradiction: positive = label in {contradicts, evolves}
  dedup:         positive = label == duplicate
NLI decision signals: contradiction = max-direction P(contra);
                      dedup = min-direction P(entail) (bidirectional entailment).

Output: results.json (raw scores) + report.md (metric tables).
Usage: /root/venvs/ai/bin/python bench.py
Deps: torch(cpu), transformers, sentencepiece, fastembed (venv ai)
"""

import hashlib
import json
import math
import os
import sys
import time

import numpy as np

NLI_MODELS = {
    "mdeberta-xnli": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "deberta-v3-mnli-fever-anli": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "nli-deberta-v3-base": "cross-encoder/nli-deberta-v3-base",
    "roberta-large-mnli": "FacebookAI/roberta-large-mnli",
}
ENGLISH_ONLY = {"deberta-v3-mnli-fever-anli", "nli-deberta-v3-base", "roberta-large-mnli"}

CONTRA_POS = {"contradicts", "evolves"}
DUP_POS = {"duplicate"}


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()[:24]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def load_pairs():
    pairs = []
    for fn in ("pairs.jsonl", "synth_pairs.jsonl"):
        if os.path.exists(fn):
            for line in open(fn, encoding="utf-8"):
                pairs.append(json.loads(line))
    labels = json.load(open("labels.json"))
    out = []
    for p in pairs:
        pid = p["pair_id"]
        lab = p.get("label") or labels.get(pid, {}).get("label")
        if not lab:
            continue
        p["label"] = lab
        out.append(p)
    return out


def english(text, tr):
    return tr.get(sha(text), text)


class NliScorer:
    def __init__(self, model_id):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()
        id2label = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.idx = {}
        for i, l in id2label.items():
            if "entail" in l:
                self.idx["entail"] = i
            elif "contra" in l:
                self.idx["contra"] = i

    def score(self, premise, hypothesis):
        with self.torch.no_grad():
            enc = self.tok(premise, hypothesis, return_tensors="pt",
                           truncation=True, max_length=512)
            probs = self.torch.softmax(self.model(**enc).logits[0], dim=-1)
        return float(probs[self.idx["entail"]]), float(probs[self.idx["contra"]])


def metrics(y, s):
    """AUC + best-F1 sweep + precision/recall at best-F1 threshold."""
    y = np.array(y, dtype=bool)
    s = np.array(s, dtype=float)
    if y.all() or (~y).any() is False or not y.any():
        return {"auc": None, "f1": None, "note": "single-class"}
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    best = {"f1": -1.0}
    for t in sorted(set(s)):
        pred = s >= t
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best["f1"]:
            best = {"f1": f1, "threshold": float(t), "precision": prec,
                    "recall": rec, "fp": fp, "fn": fn}
    return {"auc": round(float(auc), 4), **{k: (round(v, 4) if isinstance(v, float) else v)
                                            for k, v in best.items()}}


def main():
    pairs = load_pairs()
    tr = json.load(open("translations.json")) if os.path.exists("translations.json") else {}
    print(f"{len(pairs)} labeled pairs; {len(tr)} translations", file=sys.stderr)

    results = {p["pair_id"]: {"label": p["label"], "source": p.get("source", p.get("family"))}
               for p in pairs}
    latency = {}

    # --- vec + jina via mnemos (native arm and english arm) ---
    sys.path.insert(0, "/root/work/mnemos")
    from mnemos.embed import embed
    from mnemos.rerank import rerank

    for arm in ("native", "english"):
        get = (lambda t: t) if arm == "native" else (lambda t: english(t, tr))
        t0 = time.time()
        texts = []
        for p in pairs:
            texts += [get(p["a_content"]), get(p["b_content"])]
        vecs = embed(texts, prefix="passage")
        vecs = np.array(vecs)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        for i, p in enumerate(pairs):
            cos = float(vecs[2 * i] @ vecs[2 * i + 1])
            results[p["pair_id"]][f"vec:{arm}"] = round(cos, 4)
        latency[f"vec:{arm}"] = (time.time() - t0) / len(pairs)

        t0 = time.time()
        for p in pairs:
            a, b = get(p["a_content"]), get(p["b_content"])
            s1 = rerank(a, [{"id": 1, "text": b}])[0].get("_rerank_score", 0.0)
            s2 = rerank(b, [{"id": 1, "text": a}])[0].get("_rerank_score", 0.0)
            results[p["pair_id"]][f"jina:{arm}"] = round(
                max(sigmoid(s1), sigmoid(s2)), 4)
        latency[f"jina:{arm}"] = (time.time() - t0) / len(pairs)

    # --- NLI models ---
    for name, model_id in NLI_MODELS.items():
        print(f"loading {name}...", file=sys.stderr)
        scorer = NliScorer(model_id)
        arms = ("english",) if name in ENGLISH_ONLY else ("native", "english")
        for arm in arms:
            get = (lambda t: t) if arm == "native" else (lambda t: english(t, tr))
            t0 = time.time()
            for p in pairs:
                a, b = get(p["a_content"]), get(p["b_content"])
                e_ab, c_ab = scorer.score(a, b)
                e_ba, c_ba = scorer.score(b, a)
                r = results[p["pair_id"]]
                r[f"{name}:contra:{arm}"] = round(max(c_ab, c_ba), 4)
                r[f"{name}:bient:{arm}"] = round(min(e_ab, e_ba), 4)
            latency[f"{name}:{arm}"] = (time.time() - t0) / len(pairs) / 2
        del scorer

    json.dump({"results": results, "latency_s_per_pair": latency},
              open("results.json", "w"), indent=1)

    # --- report ---
    labels = [results[p["pair_id"]]["label"] for p in pairs]
    y_contra = [l in CONTRA_POS for l in labels]
    y_dup = [l in DUP_POS for l in labels]

    lines = ["# NLI bench results", "",
             f"pairs: {len(pairs)} (label distribution: " +
             ", ".join(f"{l}={labels.count(l)}" for l in sorted(set(labels))) + ")", ""]
    for task, y in (("contradiction (pos=contradicts+evolves)", y_contra),
                    ("dedup (pos=duplicate)", y_dup)):
        lines += [f"## {task}", "",
                  "| scorer | arm | AUC | bestF1 | prec | rec | thr | fp | fn |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for key in sorted({k for r in results.values() for k in r
                           if ":" in k and not k.startswith(("label", "source"))}):
            if "contradiction" in task:
                if not (key.startswith(("vec:", "jina:")) or ":contra:" in key):
                    continue
            else:
                if not (key.startswith(("vec:", "jina:")) or ":bient:" in key):
                    continue
            s = [results[p["pair_id"]].get(key) for p in pairs]
            if any(v is None for v in s):
                continue
            m = metrics(y, s)
            scorer_name, arm = key.rsplit(":", 1)
            lines.append(
                f"| {scorer_name} | {arm} | {m.get('auc')} | {m.get('f1')} | "
                f"{m.get('precision')} | {m.get('recall')} | {m.get('threshold')} | "
                f"{m.get('fp')} | {m.get('fn')} |")
        lines.append("")
    lines += ["## latency (s/pair, CPU)", ""] + [
        f"- {k}: {v:.3f}" for k, v in sorted(latency.items())]
    open("report.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
