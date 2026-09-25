#!/usr/bin/env bash
# One-shot, resumable, logged run of the full pipeline on a fresh Linux box.
#
#   bash run.sh                       # everything: venv, data download, tests, pipeline, summary
#   bash run.sh --skip-download       # data already in $DATA
#   bash run.sh --from train          # re-run from a step (steps listed below), keeping earlier outputs
#   bash run.sh --fresh               # ignore .done markers and redo every step
#
# Environment overrides (all optional):
#   DATA=dataset  WORK=work  OUT=output  LOGS=logs  ROUNDS1=150  ROUNDS2=100  PY=python3
#
# Every step writes logs/<nn>_<step>.log, prints its wall time, and leaves $WORK/.done/<step>
# so a crashed or interrupted run continues where it stopped. Run it under nohup or tmux:
#   nohup bash run.sh > logs/run.out 2>&1 &   ;   tail -f logs/run.out
set -euo pipefail
cd "$(dirname "$0")"

DATA="${DATA:-dataset}"; WORK="${WORK:-work}"; OUT="${OUT:-output}"; LOGS="${LOGS:-logs}"
ROUNDS1="${ROUNDS1:-150}"; ROUNDS2="${ROUNDS2:-100}"; PY="${PY:-python3}"
LFS_BASE="https://media.githubusercontent.com/media/SukhvirKooner/ml-challenge-2026/main"
SKIP_DOWNLOAD=0; FROM=""; FRESH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-download) SKIP_DOWNLOAD=1 ;;
    --from) FROM="$2"; shift ;;
    --fresh) FRESH=1 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac; shift
done
mkdir -p "$LOGS" "$WORK/.done" "$OUT"
T_START=$(date +%s)
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOGS/run.log"; }
STEPS=(venv download tests folds lexicon prepare_train blocking_train features_train prepare_test blocking_test features_test train leakage_check select_threshold predict validate summary)
N=0
step() {  # step <name> <command...>
  local name="$1"; shift
  N=$((N+1)); local tag; tag=$(printf '%02d_%s' "$N" "$name")
  if [ -n "$FROM" ]; then
    local skip=1; for s in "${STEPS[@]}"; do [ "$s" = "$FROM" ] && skip=0; [ "$s" = "$name" ] && break; done
    if [ $skip -eq 1 ] && [ -f "$WORK/.done/$name" ]; then log "skip  $tag (before --from $FROM)"; return; fi
  fi
  if [ $FRESH -eq 0 ] && [ -z "$FROM" ] && [ -f "$WORK/.done/$name" ]; then log "skip  $tag (done earlier; --fresh to redo)"; return; fi
  log "start $tag: $*"
  local t0; t0=$(date +%s)
  if "$@" > "$LOGS/$tag.log" 2>&1; then
    touch "$WORK/.done/$name"
    log "done  $tag in $(( $(date +%s) - t0 ))s"
  else
    log "FAIL  $tag after $(( $(date +%s) - t0 ))s -- see $LOGS/$tag.log (last lines below)"
    tail -n 25 "$LOGS/$tag.log" | tee -a "$LOGS/run.log"
    exit 1
  fi
}

# ---------------------------------------------------------------- environment
setup_venv() {
  if [ ! -x .venv/bin/python ]; then "$PY" -m venv .venv; fi
  . .venv/bin/activate
  pip install -q --upgrade pip
  # pinned versions target Python 3.14; fall back to the latest compatible releases on older Pythons
  pip install -q -r requirements.txt || pip install -q lightgbm numpy polars pyarrow rapidfuzz scipy unidecode
  pip install -q pytest
  python -c "import lightgbm, polars, rapidfuzz, scipy, unidecode, pyarrow; print('python', __import__('sys').version.split()[0], 'polars', polars.__version__, 'lightgbm', lightgbm.__version__)"
  echo "cpus: $(nproc)  mem: $(free -g | awk '/Mem/{print $2}') GB"
}
download() {
  mkdir -p "$DATA/train" "$DATA/test"
  for f in train_ground_truth train_source1 train_source2 train_source3; do
    [ -s "$DATA/train/$f.tsv" ] || curl -fSL --retry 5 -o "$DATA/train/$f.tsv" "$LFS_BASE/$f.tsv"
  done
  for f in test_source1 test_source2 test_source3; do
    [ -s "$DATA/test/$f.tsv" ] || curl -fSL --retry 5 -o "$DATA/test/$f.tsv" "$LFS_BASE/$f.tsv"
  done
  ls -la "$DATA/train" "$DATA/test"; wc -l "$DATA"/*/*.tsv
}
summary() {
  python tools/summarize_reports.py --work-dir "$WORK" | tee "$WORK/summary.md"
  echo; echo "submission files:"; ls -la "$OUT"
}

step venv setup_venv
. .venv/bin/activate
if [ $SKIP_DOWNLOAD -eq 0 ]; then step download download; else N=$((N+1)); log "skip  02_download (--skip-download)"; fi
step tests python -m pytest tests -q
D="$DATA"; W="$WORK"
step folds            python src/folds.py            --data-dir "$D" --work-dir "$W"
step lexicon          python src/build_lexicon.py    --data-dir "$D" --work-dir "$W"
step prepare_train    python src/prepare.py          --data-dir "$D" --work-dir "$W" --split train
step blocking_train   python src/blocking.py         --data-dir "$D" --work-dir "$W" --split train
step features_train   python src/pair_features.py    --data-dir "$D" --work-dir "$W" --split train
step prepare_test     python src/prepare.py          --data-dir "$D" --work-dir "$W" --split test
step blocking_test    python src/blocking.py         --data-dir "$D" --work-dir "$W" --split test
step features_test    python src/pair_features.py    --data-dir "$D" --work-dir "$W" --split test
step train            python src/train.py            --data-dir "$D" --work-dir "$W" --rounds1 "$ROUNDS1" --rounds2 "$ROUNDS2"
step leakage_check    python src/leakage_check.py    --data-dir "$D" --work-dir "$W"
step select_threshold python src/select_threshold.py --data-dir "$D" --work-dir "$W"
step predict          python src/predict.py          --data-dir "$D" --work-dir "$W" --out-dir "$OUT"
step validate         python src/validate_submission.py --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" --test-dir "$D/test"
step summary          summary
log "ALL DONE in $(( ($(date +%s) - T_START) / 60 )) min. Submission: $OUT/matching_results.tsv + $OUT/candidate_pairs.tsv; diagnostics: $WORK/summary.md, $WORK/validation_report.json, $WORK/profile.json"
