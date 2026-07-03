#!/usr/bin/env python3
"""Export the NLI models to ONNX (fp32) for the mnemos NLI decision layer.

Produces the layout mnemos.nli expects (MNEMOS_NLI_ONNX_DIR, default
~/.cache/mnemos/nli-onnx):
    <dir>/en/model.onnx     English checkpoint
    <dir>/multi/model.onnx  multilingual checkpoint
plus tokenizer/config files alongside each model.

fp32 on purpose: int8 dynamic quantization was validated against the
nli-bench pairs and REJECTED. It collapses DeBERTa-v3 scoring to chance
(contradiction AUC 0.94 -> 0.51 English, 0.84 -> 0.48 multilingual, dozens
of threshold flips); the disentangled-attention architecture does not
survive dynamic weight quantization. fp32 ONNX is score-identical to the
torch checkpoints (max probability drift 1e-05, zero threshold flips on
114 bench pairs). Do not re-add quantization without re-running the
parity gate.

Run this ONCE per machine, or export on one machine and copy the output
directory to the others. Export needs the heavy tooling
(mnemos[nli-export]: torch, optimum-onnx); runtime needs none of it.

Usage: python scripts/export_nli_onnx.py [output-dir]
"""

import os
import shutil
import subprocess
import sys
import tempfile

from mnemos.constants import NLI_EN_MODEL, NLI_MULTI_MODEL


def export_one(model_id, out_dir):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, "-m", "optimum.exporters.onnx",
             "--model", model_id, "--task", "text-classification", tmp],
            check=True)
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(tmp):
            shutil.copy(os.path.join(tmp, f), os.path.join(out_dir, f))
    size = os.path.getsize(os.path.join(out_dir, "model.onnx")) / 1e6
    print(f"{model_id} -> {out_dir} ({size:.0f} MB)")


def smoke(base):
    os.environ["MNEMOS_NLI_ONNX_DIR"] = base
    os.environ["MNEMOS_NLI_BACKEND"] = "onnx"
    import mnemos.nli as nli
    nli._scorers = {}
    p = nli.p_contradiction("the API listens on port 8080",
                            "the API listens on port 9090")
    print(f"smoke P(contra) on port conflict: {p}")
    if p is None or p < 0.5:
        sys.exit("smoke check FAILED: exported models do not score sanely")


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "MNEMOS_NLI_ONNX_DIR", os.path.expanduser("~/.cache/mnemos/nli-onnx"))
    export_one(NLI_EN_MODEL, os.path.join(base, "en"))
    export_one(NLI_MULTI_MODEL, os.path.join(base, "multi"))
    smoke(base)
    print(f"done; models under {base} (the default search path)")


if __name__ == "__main__":
    main()
