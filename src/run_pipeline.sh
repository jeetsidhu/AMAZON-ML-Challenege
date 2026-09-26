#!/usr/bin/env bash
# End-to-end: data -> normalisation -> blocking -> pair features -> training -> test inference -> output
# Usage: bash src/run_pipeline.sh <dataset_dir> <work_dir> <output_dir>
set -euo pipefail
DATA="${1:?dataset dir (contains train/ and test/)}"
WORK="${2:-work}"
OUT="${3:-output}"
SRC="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"

"$PY" "$SRC/folds.py"         --data-dir "$DATA" --work-dir "$WORK"   # folds first: preprocessing is fit per fold
"$PY" "$SRC/build_lexicon.py" --data-dir "$DATA" --work-dir "$WORK"
for SPLIT in train test; do
  "$PY" "$SRC/prepare.py"       --data-dir "$DATA" --work-dir "$WORK" --split "$SPLIT"
  "$PY" "$SRC/blocking.py"      --data-dir "$DATA" --work-dir "$WORK" --split "$SPLIT"
  "$PY" "$SRC/pair_features.py" --data-dir "$DATA" --work-dir "$WORK" --split "$SPLIT"
done
"$PY" "$SRC/train.py"            --data-dir "$DATA" --work-dir "$WORK"
"$PY" "$SRC/leakage_check.py"    --data-dir "$DATA" --work-dir "$WORK"
"$PY" "$SRC/select_threshold.py" --data-dir "$DATA" --work-dir "$WORK"
"$PY" "$SRC/tune_thresholds.py"  --data-dir "$DATA" --work-dir "$WORK" ${THRESHOLD_ARGS:-}   # global vs per-class policies; writes threshold_policy.json
"$PY" "$SRC/predict.py"          --data-dir "$DATA" --work-dir "$WORK" --out-dir "$OUT"
"$PY" "$SRC/validate_submission.py" --matching "$OUT/matching_results.tsv" \
      --candidate "$OUT/candidate_pairs.tsv" --test-dir "$DATA/test"
