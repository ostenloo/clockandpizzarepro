#!/bin/bash
# Runs after the E2 sweep finishes. Every step is resumable and idempotent.
cd ~/clockandpizzarepro
PY=.venv/bin/python
log() { echo "[queue] $* -- $(date)"; }

log "waiting for E2 to finish"
while pgrep -f "run_e2.py" > /dev/null; do sleep 60; done
log "E2 finished"

# First: restore the registry records lost when a git checkout landed on runs.jsonl
# mid-sweep. Safe only now that nothing is appending to it.
$PY -u scripts/rebuild_registry.py > results/rebuild.log 2>&1
log "registry rebuilt ($(wc -l < results/runs.jsonl) records)"

$PY -u scripts/e2.py --skip-training > results/e2.log 2>&1
log "E2 analysis exited $?"

for exp in e8 e6 e5 e9; do
  log "starting $exp"
  $PY -u scripts/$exp.py > results/$exp.log 2>&1
  log "$exp exited $?"
done

$PY -u scripts/report.py > results/report.log 2>&1
log "all done"
