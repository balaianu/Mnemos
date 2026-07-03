#!/usr/bin/env python3
"""Mechanical-merge bench (pass 2): can line-union NLI replace the MERGE LLM?

Arms (data from harvest.py):
  A. Synthetic overlap splits, exact ground truth. Metric: recovery rate
     (output line set == original line set). A missing unique line is an
     NLI false-duplicate at tau; an uncollapsed shared line is a miss of
     an exact duplicate (should be impossible via the fast path).
  B. Production high-cosine pairs. Gated first with the validated phase-2
     line-level shared-fact gate (>= 0.70); passers are mechanically
     merged. Metrics: gate pass rate, compression, full dropped-line audit
     (every drop listed with partner and score for manual review).
  C. The 2026-07-03 production clusters: mechanical merge vs the actual
     Sonnet+splitter outputs on identical inputs. Metrics per arm: input
     line coverage (an input line counts covered when some output line
     entails it at >= 0.70) and digit integrity (digit tokens present in
     inputs that survive to the output). Mechanical coverage is by
     construction except for dropped duplicates; the NLI coverage scoring
     is independent evidence only for the LLM arm (NLI judging NLI is
     circular and is labeled as such in the report).

Output: results.json + report.md.
Usage: /root/venvs/ai/bin/python bench.py [--arms arms.json]
Deps: mnemos[nli] with a local ONNX export (venv ai).
"""

import argparse
import json
import re
import time

from mnemos import nli

from mechanical import mechanical_merge, lines_of, TAU_DUP

TAU_GATE = 0.70
TAU_COVER = 0.70
GATE_MAX_LINES = 8
_DIGIT = re.compile(r"\d[\d.,:/-]*")


def gate_score(a_content, b_content):
    """Max line-level bidirectional entailment between two memories."""
    best = 0.0
    la = lines_of(a_content)[:GATE_MAX_LINES]
    lb = lines_of(b_content)[:GATE_MAX_LINES]
    for x in la:
        for y in lb:
            e = nli.bidirectional_entailment(x, y)
            if e is not None and e > best:
                best = e
    return best


def line_coverage(input_lines, output_text, tau=TAU_COVER):
    """Fraction of input lines entailed by some output line."""
    out_lines = [ln.strip() for ln in output_text.split("\n") if ln.strip()]
    misses = []
    for ln in input_lines:
        if any(ln == ol for ol in out_lines):
            continue
        covered = False
        for ol in out_lines:
            e, _ = nli._score_pair(ol, ln)[0]
            if e >= tau:
                covered = True
                break
        if not covered:
            misses.append(ln)
    n = len(input_lines)
    return (n - len(misses)) / n if n else 1.0, misses


def digit_integrity(input_texts, output_text):
    want = set()
    for t in input_texts:
        want.update(_DIGIT.findall(t))
    have = set(_DIGIT.findall(output_text))
    missing = sorted(want - have)
    return (len(want - set(missing)) / len(want) if want else 1.0), missing


def run_arm_a(cases):
    results = []
    for c in cases:
        m = mechanical_merge([c["a"], c["b"]])
        out_lines = m["merged"].split("\n") if m["merged"] else []
        truth = c["truth_lines"]
        missing = [ln for ln in truth if ln not in out_lines]
        extra_dupes = len(out_lines) - len(set(out_lines))
        uncollapsed = len(out_lines) - len(truth) + len(missing)
        results.append({
            "source_id": c["source_id"], "n_truth": len(truth),
            "missing_lines": missing, "uncollapsed": max(0, uncollapsed),
            "extra_dupes": extra_dupes,
            "exact_recovery": not missing and uncollapsed <= 0,
        })
    return results


def run_arm_b(pairs):
    results = []
    for p in pairs:
        g = gate_score(p["a"]["content"], p["b"]["content"])
        rec = {"a_id": p["a"]["id"], "b_id": p["b"]["id"],
               "cosine": p["cosine"], "gate": round(g, 3),
               "gate_pass": g >= TAU_GATE}
        if rec["gate_pass"]:
            m = mechanical_merge([p["a"], p["b"]])
            total = len(m["kept"]) + len(m["dropped"])
            rec.update({
                "lines_total": total, "lines_dropped": len(m["dropped"]),
                "compression": round(len(m["dropped"]) / total, 3) if total else 0,
                "dropped_audit": [
                    {"line": d["line"][:120], "dup_line": d["dup_line"][:120],
                     "score": d["score"]} for d in m["dropped"]],
            })
        results.append(rec)
    return results


def run_arm_c(clusters):
    results = {}
    for name, case in clusters.items():
        inputs = case["inputs"]
        input_lines = []
        for m in inputs:
            input_lines.extend(lines_of(m["content"]))
        mech = mechanical_merge(inputs)

        cov_prod, miss_prod = line_coverage(input_lines, case["prod_output_text"])
        cov_mech, miss_mech = line_coverage(input_lines, mech["merged"])
        digi_prod, dmiss_prod = digit_integrity(
            [m["content"] for m in inputs], case["prod_output_text"])
        digi_mech, dmiss_mech = digit_integrity(
            [m["content"] for m in inputs], mech["merged"])

        results[name] = {
            "n_inputs": len(inputs), "n_input_lines": len(input_lines),
            "prod": {"coverage": round(cov_prod, 3),
                     "missed_lines": [l[:120] for l in miss_prod],
                     "digit_integrity": round(digi_prod, 3),
                     "digits_missing": dmiss_prod[:20],
                     "chars": len(case["prod_output_text"])},
            "mechanical": {"coverage": round(cov_mech, 3),
                           "missed_lines": [l[:120] for l in miss_mech],
                           "digit_integrity": round(digi_mech, 3),
                           "digits_missing": dmiss_mech[:20],
                           "chars": len(mech["merged"]),
                           "dropped": len(mech["dropped"]),
                           "dropped_audit": [
                               {"line": d["line"][:120], "score": d["score"]}
                               for d in mech["dropped"]]},
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="arms.json")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--report", default="report.md")
    args = ap.parse_args()
    arms = json.load(open(args.arms))

    t0 = time.time()
    print("Arm A: synthetic overlap splits...", flush=True)
    a = run_arm_a(arms["synthetic"])
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)
    print("Arm B: production high-cosine pairs...", flush=True)
    b = run_arm_b(arms["highsim"])
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)
    print("Arm C: 2026-07-03 cluster head-to-head...", flush=True)
    c = run_arm_c(arms["clusters"])
    runtime = time.time() - t0

    a_pass = sum(1 for r in a if r["exact_recovery"])
    b_pass = [r for r in b if r["gate_pass"]]
    summary = {
        "runtime_s": round(runtime, 1),
        "armA_exact_recovery": f"{a_pass}/{len(a)}",
        "armA_false_dup_lines": sum(len(r["missing_lines"]) for r in a),
        "armB_gate_pass": f"{len(b_pass)}/{len(b)}",
        "armB_total_dropped": sum(r.get("lines_dropped", 0) for r in b_pass),
        "armC": {k: {"prod_cov": v["prod"]["coverage"],
                     "mech_cov": v["mechanical"]["coverage"],
                     "prod_digits": v["prod"]["digit_integrity"],
                     "mech_digits": v["mechanical"]["digit_integrity"]}
                 for k, v in c.items()},
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "armA": a, "armB": b, "armC": c},
                  f, indent=1, ensure_ascii=False)

    lines = [
        "# Mechanical-merge bench (pass 2)",
        "",
        f"Runtime {summary['runtime_s']}s. tau_dup={TAU_DUP}, "
        f"tau_gate={TAU_GATE}, tau_cover={TAU_COVER}.",
        "",
        "## Arm A: synthetic overlap splits (exact ground truth)",
        "",
        f"- exact recovery: **{summary['armA_exact_recovery']}**",
        f"- unique lines falsely dropped as duplicates: "
        f"**{summary['armA_false_dup_lines']}**",
        "",
    ]
    for r in a:
        if r["missing_lines"]:
            lines.append(f"- FALSE DUP in #{r['source_id']}: " +
                         "; ".join(l[:100] for l in r["missing_lines"]))
    lines += [
        "",
        "## Arm B: production high-cosine pairs",
        "",
        f"- gate pass: {summary['armB_gate_pass']} "
        f"(cosine >= {arms['highsim'][0]['cosine'] if arms['highsim'] else '?'} "
        f"down to 0.78)",
        f"- lines dropped across all passers: {summary['armB_total_dropped']}",
        "",
        "### Dropped-line audit (every drop, for manual review)",
        "",
    ]
    for r in b_pass:
        for d in r.get("dropped_audit", []):
            lines.append(f"- [{r['a_id']}x{r['b_id']} s={d['score']}] "
                         f"DROP: {d['line']}  ||  KEPT: {d['dup_line']}")
    lines += ["", "## Arm C: head-to-head on the 2026-07-03 clusters", ""]
    lines.append("| cluster | arm | line coverage | digit integrity | chars |")
    lines.append("|---|---|---|---|---|")
    for name, v in c.items():
        lines.append(f"| {name} | prod (Sonnet+splitter) | "
                     f"{v['prod']['coverage']} | "
                     f"{v['prod']['digit_integrity']} | {v['prod']['chars']} |")
        lines.append(f"| {name} | mechanical | "
                     f"{v['mechanical']['coverage']} | "
                     f"{v['mechanical']['digit_integrity']} | "
                     f"{v['mechanical']['chars']} |")
    lines += [
        "",
        "Coverage scored by NLI (output line entails input line at >= "
        f"{TAU_COVER}). For the mechanical arm this is partly circular "
        "(NLI grading NLI) and is reported for symmetry; its real "
        "guarantee is by construction. Missed-line lists in results.json "
        "are the manual review queue.",
    ]
    for name, v in c.items():
        if v["prod"]["missed_lines"]:
            lines.append(f"- {name} prod missed: " +
                         " | ".join(v["prod"]["missed_lines"][:5]))
        if v["mechanical"]["missed_lines"]:
            lines.append(f"- {name} mech missed: " +
                         " | ".join(v["mechanical"]["missed_lines"][:5]))
    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out} + {args.report}")


if __name__ == "__main__":
    main()
