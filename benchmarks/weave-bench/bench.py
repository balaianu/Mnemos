#!/usr/bin/env python3
"""Weave-without-LLM bench: can the NLI layer make Phase 3 link decisions?

Two decisions are benched against the silver labels in pairs.jsonl
(harvest.py):

  1. GATE: link or no-link. Scores every pair (typed links = positive,
     unlinked same-band pairs = negative) and reports AUC per signal.
  2. TYPE: which relation. Two methods, evaluated on typed pairs only:
       zeroshot  - 5 hypothesis templates through the NLI model, argmax
                   P(entailment). Tests "can NLI name the relation".
       features  - directional entailment + line-level contradiction with
                   fixed thresholds. Tests "can NLI make the decisions
                   production needs" (contradicts / evolves / other).

Both are also scored on the collapsed taxonomy {contradicts, evolves, other}
because informs/enables/reflects are hypothesized to be non-separable (and
possibly non-actionable: retrieval treats them identically).

Silver labels come from the weave LLM (Haiku), so agreement is measured, not
correctness. The report includes a disagreement sample for manual review.

Output: results.json (per-pair scores) + report.md (metric tables).
Usage: /root/venvs/ai/bin/python bench.py [--pairs pairs.jsonl] [--limit N]
Deps: mnemos[nli] with a local ONNX export (venv ai).
"""

import argparse
import json
import time

from mnemos import nli

TRUNC_CHARS = 700

TEMPLATES = {
    "evolves": "The second note is a newer version of the first note, updating or extending the same matter.",
    "informs": "The first note provides background that helps explain the second note.",
    "contradicts": "The two notes state conflicting facts about the same thing.",
    "enables": "The thing described in the first note makes the thing in the second note possible.",
    "reflects": "The two notes show the same underlying pattern in different situations.",
}

# features-method thresholds (fixed a priori, not fitted on this data)
TH_LINE_CONTRA = 0.80
TH_EVOLVES_ENTAIL = 0.60
COLLAPSE = {"informs": "other", "enables": "other", "reflects": "other",
            "evolves": "evolves", "contradicts": "contradicts"}


def trunc(text):
    return text[:TRUNC_CHARS]


def score_routed(premise, hypothesis):
    """Language-routed single-direction score, mirrors nli._score_pair routing."""
    multilingual = not (nli.is_english(premise) and nli.is_english(hypothesis))
    scorer = nli._get_scorer(multilingual=multilingual)
    return scorer.score(premise, hypothesis)


def score_pair(rec):
    a, b = trunc(rec["a_content"]), trunc(rec["b_content"])
    (e_ab, c_ab), (e_ba, c_ba) = nli._score_pair(a, b)
    line_contra = nli.line_max_contradiction(rec["a_content"], rec["b_content"]) or 0.0
    premise = f"NOTE ONE:\n{a}\n\nNOTE TWO:\n{b}"
    zeroshot = {}
    for label, hyp in TEMPLATES.items():
        e, _ = score_routed(premise, hyp)
        zeroshot[label] = e
    return {
        "e_ab": e_ab, "e_ba": e_ba, "c_ab": c_ab, "c_ba": c_ba,
        "line_contra": line_contra, "zeroshot": zeroshot,
    }


def predict_zeroshot(s):
    return max(s["zeroshot"], key=s["zeroshot"].get)


def predict_features(s):
    if s["line_contra"] >= TH_LINE_CONTRA:
        return "contradicts"
    if max(s["e_ab"], s["e_ba"]) >= TH_EVOLVES_ENTAIL:
        return "evolves"
    return "other"


def auc(pos_scores, neg_scores):
    """Rank-based AUC, no sklearn."""
    pairs = [(x, 1) for x in pos_scores] + [(x, 0) for x in neg_scores]
    pairs.sort(key=lambda t: t[0])
    rank_sum, n_pos, n_neg = 0.0, len(pos_scores), len(neg_scores)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def spearman(xs, ys):
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and vals[order[j]] == vals[order[i]]:
                j += 1
            avg = (i + j + 1) / 2.0
            for k in range(i, j):
                r[order[k]] = avg
            i = j
        return r
    n = len(xs)
    if n < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def confusion(y_true, y_pred, labels):
    m = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        m[t][p] += 1
    return m


def fmt_confusion(m, labels):
    head = "| true \\ pred | " + " | ".join(labels) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    rows = [head, sep]
    for t in labels:
        rows.append("| " + t + " | " + " | ".join(str(m[t][p]) for p in labels) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--report", default="report.md")
    args = ap.parse_args()

    records = [json.loads(ln) for ln in open(args.pairs)]
    if args.limit:
        records = records[:args.limit]

    t0 = time.time()
    for i, rec in enumerate(records):
        rec["scores"] = score_pair(rec)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(records)} scored ({el:.0f}s, "
                  f"{el/(i+1)*1000:.0f}ms/pair)", flush=True)
    runtime = time.time() - t0

    typed = [r for r in records if r["label"] != "none"]
    negs = [r for r in records if r["label"] == "none"]

    # GATE: AUC per candidate signal
    def gsig(r, name):
        s = r["scores"]
        if name == "zs_max":
            return max(s["zeroshot"].values())
        if name == "ent_max":
            return max(s["e_ab"], s["e_ba"])
        if name == "line_contra":
            return s["line_contra"]
        if name == "combo":
            return max(max(s["zeroshot"].values()), max(s["e_ab"], s["e_ba"]),
                       s["line_contra"])
        raise KeyError(name)

    gate = {}
    for name in ("zs_max", "ent_max", "line_contra", "combo"):
        gate[name] = auc([gsig(r, name) for r in typed],
                         [gsig(r, name) for r in negs])

    # TYPE: 5-way zeroshot on typed pairs
    y_true5 = [r["label"] for r in typed]
    y_zs5 = [predict_zeroshot(r["scores"]) for r in typed]
    labels5 = list(TEMPLATES.keys())
    acc5 = sum(t == p for t, p in zip(y_true5, y_zs5)) / len(typed)

    # TYPE: collapsed 3-way, both methods
    y_true3 = [COLLAPSE[t] for t in y_true5]
    y_zs3 = [COLLAPSE[p] for p in y_zs5]
    y_ft3 = [predict_features(r["scores"]) for r in typed]
    labels3 = ["contradicts", "evolves", "other"]
    acc3_zs = sum(t == p for t, p in zip(y_true3, y_zs3)) / len(typed)
    acc3_ft = sum(t == p for t, p in zip(y_true3, y_ft3)) / len(typed)
    majority3 = max(y_true3.count(l) for l in labels3) / len(typed)

    # strength correlation
    strengths = [(max(r["scores"]["zeroshot"].values()), r["silver_strength"])
                 for r in typed if r["silver_strength"] is not None]
    rho = spearman([x for x, _ in strengths], [y for _, y in strengths])

    # disagreement sample for manual review
    disagreements = [
        {"a_id": r["a_id"], "b_id": r["b_id"], "silver": r["label"],
         "zeroshot": p, "probs": {k: round(v, 3) for k, v in
                                  r["scores"]["zeroshot"].items()}}
        for r, p in zip(typed, y_zs5) if r["label"] != p
    ][:12]

    results = {
        "n_typed": len(typed), "n_negatives": len(negs),
        "runtime_s": round(runtime, 1),
        "ms_per_pair": round(runtime / len(records) * 1000),
        "gate_auc": {k: round(v, 4) for k, v in gate.items()},
        "type_acc_5way_zeroshot": round(acc5, 4),
        "type_acc_3way_zeroshot": round(acc3_zs, 4),
        "type_acc_3way_features": round(acc3_ft, 4),
        "majority_baseline_3way": round(majority3, 4),
        "strength_spearman": round(rho, 4),
        "confusion_5way_zeroshot": confusion(y_true5, y_zs5, labels5),
        "confusion_3way_features": confusion(y_true3, y_ft3, labels3),
        "disagreement_sample": disagreements,
        "pairs": [{"a_id": r["a_id"], "b_id": r["b_id"], "label": r["label"],
                   "scores": r["scores"]} for r in records],
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    label_counts = {}
    for r in typed:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    lines = [
        "# Weave-without-LLM bench",
        "",
        f"{len(typed)} typed weave pairs (silver labels from the production "
        f"weave LLM), {len(negs)} unlinked same-band negatives. "
        f"{results['ms_per_pair']}ms/pair, total {results['runtime_s']}s.",
        "",
        f"Class balance: {json.dumps(label_counts, sort_keys=True)}. "
        "Small-n classes (contradicts, enables, evolves) carry wide error "
        "bars; treat per-class numbers as directional.",
        "",
        "## Gate (link / no-link), AUC",
        "",
        "| signal | AUC |",
        "|---|---|",
    ]
    for k, v in results["gate_auc"].items():
        lines.append(f"| {k} | {v:.3f} |")
    lines += [
        "",
        "## Relation typing",
        "",
        f"- 5-way zeroshot accuracy: **{acc5:.3f}**",
        f"- 3-way collapsed zeroshot accuracy: **{acc3_zs:.3f}**",
        f"- 3-way features accuracy: **{acc3_ft:.3f}**",
        f"- 3-way majority baseline: {majority3:.3f}",
        f"- strength Spearman (zs max prob vs silver): {rho:.3f}",
        "",
        "### 5-way confusion (zeroshot)",
        "",
        fmt_confusion(results["confusion_5way_zeroshot"], labels5),
        "",
        "### 3-way confusion (features)",
        "",
        fmt_confusion(results["confusion_3way_features"], labels3),
        "",
        "### Disagreement sample (manual review queue)",
        "",
    ]
    for d in disagreements:
        lines.append(f"- #{d['a_id']} vs #{d['b_id']}: silver={d['silver']} "
                     f"zeroshot={d['zeroshot']} probs={d['probs']}")
    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("pairs", "confusion_5way_zeroshot",
                                   "confusion_3way_features",
                                   "disagreement_sample")}, indent=2))
    print(f"wrote {args.out} + {args.report}")


if __name__ == "__main__":
    main()
