#!/bin/bash
# Everything still outstanding, in cheapest-and-highest-priority-first order.
# Waits for the first queue (E5 -> E9 -> report) to finish, then:
#   e8  ~4 min   re-run with the denser checkpoint grid, fixes the App. K analysis
#   e3  ~8.5 h   width sweep -> T14 (the last P1 target)
#   e7  ~6.6 h   setup variants -> App. I figures
#   e4  ~21 h    depth sweep -> T15
# Every script is resumable; killing this loses at most the ensemble in flight.
cd ~/clockandpizzarepro
PY=.venv/bin/python
log() { echo "[rest] $* -- $(date)"; }

log "waiting for the first queue to finish"
until grep -q "all done" results/queue.log 2>/dev/null; do sleep 60; done
log "first queue done"

log "starting e8 (re-run, denser checkpoints)"
$PY -u scripts/e8.py > results/e8.log 2>&1; log "e8 exited $?"

log "starting e3 (width sweep, T14)"
$PY -u scripts/e3.py --budget-gb 20 > results/e3.log 2>&1; log "e3 exited $?"

log "starting e7 (setup variants)"
$PY -u scripts/e7.py --chunk 64 > results/e7.log 2>&1; log "e7 exited $?"

log "starting e4 (depth sweep, T15)"
$PY -u scripts/e4.py --chunk 64 > results/e4.log 2>&1; log "e4 exited $?"

$PY -u scripts/report.py > results/report.log 2>&1
log "ALL EXPERIMENTS COMPLETE"
