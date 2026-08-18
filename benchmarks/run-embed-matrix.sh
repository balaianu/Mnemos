#!/bin/bash
# Embedder x reranker matrix on LongMemEval (full 500 questions).
# Sequential, cheapest model first so partial results arrive early.
# Arena guard on: a 20h run with a grow-only ONNX arena would blow up RSS.
set -u
cd /root/work/mnemos/benchmarks
PY=/root/venvs/ai/bin/python
export MNEMOS_DISABLE_MEM_ARENA=1
run () {
  local model=$1 dims=$2 tag=$3
  echo "=== $(date '+%F %T') starting $tag ($model, $dims dims) ==="
  MNEMOS_EMBED_MODEL=$model MNEMOS_EMBED_DIMS=$dims \
    $PY longmemeval_matrix.py --modes hybrid,hybrid+rerank --sample 180 \
      --tempdb /tmp/lme_$tag.db --out matrix_$tag.json
  echo "=== $(date '+%F %T') finished $tag ==="
}
run BAAI/bge-small-en-v1.5 384 bge-small
run BAAI/bge-base-en-v1.5  768 bge-base
run intfloat/multilingual-e5-large 1024 e5-large
echo "=== $(date '+%F %T') MATRIX COMPLETE ==="
