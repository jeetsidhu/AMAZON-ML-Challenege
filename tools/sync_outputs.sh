#!/usr/bin/env bash
# Copies the small, shareable outputs of a run into runs/<run-id>/ and pushes them to GitHub.
# Called by run.sh after every finished step when POST_STEP_CMD is set (see run.sh), or by hand:
#   bash tools/sync_outputs.sh <run-id> [work-dir] [output-dir] [logs-dir]
# What is synced: logs/, every *.json / *.md / *.txt / *.csv directly under work/ and work/train/,
# the trained models and calibration, and the submission files gzipped. Parquet / npy intermediates
# (gigabytes) and anything above 95 MB are never added. Commits only when something changed.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN_ID="${1:?run id}"; WORK="${2:-work}"; OUT="${3:-output}"; LOGS="${4:-logs}"
DEST="runs/$RUN_ID"; mkdir -p "$DEST/logs" "$DEST/work" "$DEST/work/train" "$DEST/output"
cp -u "$LOGS"/*.log "$LOGS"/*.out "$DEST/logs/" 2>/dev/null
for f in "$WORK"/*.json "$WORK"/*.md "$WORK"/*.txt "$WORK"/*.csv "$WORK"/lexicon.json; do [ -f "$f" ] && cp -u "$f" "$DEST/work/"; done
for f in "$WORK"/train/*.json; do [ -f "$f" ] && cp -u "$f" "$DEST/work/train/"; done
for f in "$OUT"/*.tsv; do
  [ -f "$f" ] || continue
  g="$DEST/output/$(basename "$f").gz"
  if [ ! -f "$g" ] || [ "$f" -nt "$g" ]; then gzip -c "$f" > "$g"; fi
done
# never stage a file GitHub would reject
find "$DEST" -type f -size +95M -print -delete | sed 's/^/dropped (too large for GitHub): /'
git add -A "$DEST" >/dev/null 2>&1
if git diff --cached --quiet; then echo "sync: nothing new"; exit 0; fi
git -c user.name="${GIT_USER:-pipeline}" -c user.email="${GIT_EMAIL:-pipeline@localhost}" commit -q -m "run $RUN_ID: outputs after $(date '+%F %T')" \
  && (git push -q origin HEAD 2>&1 || git push -q -u origin "$(git rev-parse --abbrev-ref HEAD)" 2>&1) \
  && echo "sync: pushed $DEST" || echo "sync: push failed (will retry after the next step)"
