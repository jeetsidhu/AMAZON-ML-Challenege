#!/usr/bin/env bash
# One-shot, resumable, logged run of the full pipeline on a fresh Linux box.
#
#   bash run.sh                       # everything: venv, data download, tests, pipeline, summary
#   bash run.sh --skip-download       # data already in $DATA
#   bash run.sh --from train          # re-run from a step (steps listed below), keeping earlier outputs
#   bash run.sh --fresh               # ignore .done markers and redo every step
#   bash run.sh --no-venv             # use the current Python (Kaggle / Colab: packages preinstalled)
#   bash run.sh --smoke-only          # only the smoke test: whole pipeline on a 0.3 % slice of the data (~2 min)
#   bash run.sh --skip-smoke          # skip the smoke test
#
# Environment overrides (all optional):
#   DATA=dataset  WORK=work  OUT=output  LOGS=logs  ROUNDS1=150  ROUNDS2=100  PY=python3
#   CKPT=<name>                                    # checkpoint name for train.py (default: timestamped)
#   THRESHOLD_ARGS="--density per_class --select country"   # extra arguments for tune_thresholds.py
#   POST_STEP_CMD="bash tools/sync_outputs.sh my-run"   # run after every finished step (e.g. push outputs)
#
# Every step writes logs/<nn>_<step>.log, prints its wall time, and leaves $WORK/.done/<step>
# so a crashed or interrupted run continues where it stopped. Run it under nohup or tmux:
#   nohup bash run.sh > logs/run.out 2>&1 &   ;   tail -f logs/run.out
set -euo pipefail
cd "$(dirname "$0")"
trap 'echo "[$(date "+%F %T")] ABORTED at line $LINENO: $BASH_COMMAND (exit $?)" | tee -a "${LOGS:-logs}/run.log"' ERR

DATA="${DATA:-dataset}"; WORK="${WORK:-work}"; OUT="${OUT:-output}"; LOGS="${LOGS:-logs}"
ROUNDS1="${ROUNDS1:-150}"; ROUNDS2="${ROUNDS2:-100}"; PY="${PY:-python3}"
LFS_BASE="https://media.githubusercontent.com/media/SukhvirKooner/ml-challenge-2026/main"
SKIP_DOWNLOAD=0; FROM=""; FRESH=0; NO_VENV=0; SMOKE_ONLY=0; SKIP_SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-download) SKIP_DOWNLOAD=1 ;;
    --no-venv) NO_VENV=1 ;;
    --smoke-only) SMOKE_ONLY=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --from) FROM="$2"; shift ;;
    --fresh) FRESH=1 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac; shift
done
mkdir -p "$LOGS" "$WORK/.done" "$OUT"
T_START=$(date +%s)
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOGS/run.log"; }
STEPS=(venv download tests smoke folds lexicon prepare_train blocking_train features_train prepare_test blocking_test features_test train leakage_check select_threshold tune_thresholds predict validate summary)
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
  # subshell with errexit: a failing command inside a step function fails the step
  if ( set -e; "$@" ) > "$LOGS/$tag.log" 2>&1; then
    touch "$WORK/.done/$name"
    log "done  $tag in $(( $(date +%s) - t0 ))s"
    if [ -n "${POST_STEP_CMD:-}" ]; then
      bash -c "$POST_STEP_CMD" >> "$LOGS/post_step.log" 2>&1 && log "post-step hook ok" || log "post-step hook FAILED (see $LOGS/post_step.log); continuing"
    fi
  else
    log "FAIL  $tag after $(( $(date +%s) - t0 ))s -- see $LOGS/$tag.log (last lines below)"
    tail -n 25 "$LOGS/$tag.log" | tee -a "$LOGS/run.log"
    exit 1
  fi
}

# ---------------------------------------------------------------- environment
setup_venv() {
  if [ $NO_VENV -eq 1 ]; then
    # Kaggle / Colab: keep the preinstalled stack, add only what is missing
    "$PY" -c "import lightgbm, polars, rapidfuzz, scipy, pyarrow, pytest" 2>/dev/null \
      || "$PY" -m pip install -q lightgbm polars pyarrow rapidfuzz scipy pytest
    "$PY" -c "import unidecode" 2>/dev/null || "$PY" -m pip install -q unidecode \
      || echo "WARNING: unidecode not installable (no internet?); using the accent-folding fallback"
    "$PY" -c "import polars; v=tuple(int(x) for x in polars.__version__.split('.')[:2]); assert v >= (1, 20), polars.__version__" \
      || { echo "polars is too old for this code (need >= 1.20): pip install -U polars"; exit 1; }
  else
    # A managed Python 3.12 via uv, independent of the system Python (old distros ship 3.8 whose
    # package range lacks the polars / numpy features this code uses).
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv >/dev/null; then
      echo "installing uv ..."
      curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || { echo "cannot install uv (no internet?)"; exit 1; }
      export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi
    if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null; then
      rm -rf .venv
      uv venv -q .venv --python 3.12
    fi
    uv pip install -q --python .venv/bin/python -r requirements.txt pytest \
      || uv pip install -q --python .venv/bin/python lightgbm numpy polars pyarrow rapidfuzz scipy unidecode pytest
    set +u; . .venv/bin/activate; set -u
  fi
  python -c "import lightgbm, polars, rapidfuzz, scipy, pyarrow, numpy; print('python', __import__('sys').version.split()[0], 'polars', polars.__version__, 'numpy', numpy.__version__, 'lightgbm', lightgbm.__version__, 'rapidfuzz', rapidfuzz.__version__)"
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
smoke() {
  # the whole pipeline on a 0.3 % name-group slice of the training data (+ a disjoint 0.2 % slice as a
  # labelled hold-out), tiny model settings: catches environment / API problems in ~2 minutes
  local SD="$WORK/smoke/data" SW="$WORK/smoke/work" SO="$WORK/smoke/out"
  rm -rf "$WORK/smoke"; mkdir -p "$SD" "$SW" "$SO"
  python tools/make_subset.py --data-dir "$DATA" --out-dir "$SD" --frac 0.003 --test-frac 0.002
  python src/folds.py            --data-dir "$SD" --work-dir "$SW"
  python src/build_lexicon.py    --data-dir "$SD" --work-dir "$SW"
  for sp in train test; do
    python src/prepare.py        --data-dir "$SD" --work-dir "$SW" --split $sp
    python src/blocking.py       --data-dir "$SD" --work-dir "$SW" --split $sp
    python src/pair_features.py  --data-dir "$SD" --work-dir "$SW" --split $sp
  done
  python src/train.py            --data-dir "$SD" --work-dir "$SW" --rounds1 30 --rounds2 20 --checkpoint-name smoke
  python src/leakage_check.py    --data-dir "$SD" --work-dir "$SW" --canary-rows 50000 --canary-rounds 10
  python src/select_threshold.py --data-dir "$SD" --work-dir "$SW"
  python src/tune_thresholds.py  --data-dir "$SD" --work-dir "$SW" --min-support 20
  python src/predict.py          --data-dir "$SD" --work-dir "$SW" --out-dir "$SO"
  python tools/policy_holdout_eval.py --data-dir "$SD" --work-dir "$SW" --out-dir "$SO/policies"
  python src/validate_submission.py --matching "$SO/matching_results.tsv" --candidate "$SO/candidate_pairs.tsv" --test-dir "$SD/test"
  python src/evaluate.py --pred "$SO/matching_results.tsv" --truth "$SD/test/subset_ground_truth.tsv" --out "$SW/smoke_eval.json" \
    | python -c "import json,sys; d=json.load(sys.stdin)['overall']; print('SMOKE hold-out macro F0.5 %.4f  precision %.4f  recall %.4f' % (d['macro_f05'], d['micro_precision'], d['micro_recall']))"
  echo "smoke test OK: every stage ran end to end in this environment"
}
summary() {
  python tools/summarize_reports.py --work-dir "$WORK" | tee "$WORK/summary.md"
  echo; echo "submission files:"; ls -la "$OUT"
}

step venv setup_venv
if [ $NO_VENV -eq 0 ]; then
  [ -f .venv/bin/activate ] || { log "ABORTED: .venv was not created (see $LOGS/01_venv.log)"; exit 1; }
  set +u; . .venv/bin/activate; set -u
else python() { "$PY" "$@"; }; export -f python 2>/dev/null || true; fi
if [ $SKIP_DOWNLOAD -eq 0 ]; then step download download; else N=$((N+1)); log "skip  02_download (--skip-download)"; fi
step tests python -m pytest tests -q
if [ $SKIP_SMOKE -eq 0 ]; then step smoke smoke; else N=$((N+1)); log "skip  04_smoke (--skip-smoke)"; fi
if [ $SMOKE_ONLY -eq 1 ]; then log "SMOKE ONLY: done in $(( ($(date +%s) - T_START) / 60 )) min; rerun without --smoke-only for the full pipeline (the smoke step is then skipped)"; exit 0; fi
D="$DATA"; W="$WORK"
step folds            python src/folds.py            --data-dir "$D" --work-dir "$W"
step lexicon          python src/build_lexicon.py    --data-dir "$D" --work-dir "$W"
step prepare_train    python src/prepare.py          --data-dir "$D" --work-dir "$W" --split train
step blocking_train   python src/blocking.py         --data-dir "$D" --work-dir "$W" --split train
step features_train   python src/pair_features.py    --data-dir "$D" --work-dir "$W" --split train
step prepare_test     python src/prepare.py          --data-dir "$D" --work-dir "$W" --split test
step blocking_test    python src/blocking.py         --data-dir "$D" --work-dir "$W" --split test
step features_test    python src/pair_features.py    --data-dir "$D" --work-dir "$W" --split test
step train            python src/train.py            --data-dir "$D" --work-dir "$W" --rounds1 "$ROUNDS1" --rounds2 "$ROUNDS2" ${CKPT:+--checkpoint-name "$CKPT" --resume}
step leakage_check    python src/leakage_check.py    --data-dir "$D" --work-dir "$W"
step select_threshold python src/select_threshold.py --data-dir "$D" --work-dir "$W"
step tune_thresholds  python src/tune_thresholds.py  --data-dir "$D" --work-dir "$W" ${THRESHOLD_ARGS:-}
step predict          python src/predict.py          --data-dir "$D" --work-dir "$W" --out-dir "$OUT"
step validate         python src/validate_submission.py --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" --test-dir "$D/test"
step summary          summary
log "ALL DONE in $(( ($(date +%s) - T_START) / 60 )) min. Submission: $OUT/matching_results.tsv + $OUT/candidate_pairs.tsv; diagnostics: $WORK/summary.md, $WORK/validation_report.json, $WORK/threshold_experiments.md, $WORK/profile.json; checkpoint: $WORK/checkpoints/$(cat "$WORK/checkpoints/LATEST" 2>/dev/null)"
