#!/bin/bash
# Overnight LOCAL multimodel run (MPS). Durable local writes + watchdog + resume + OOM-bail.
cd /Users/lipicsbarna/git/activation_steering
export HF_HOME=/Volumes/CsutiStorage01/hf_cache
mkdir -p "$HF_HOME"
OUT=results/multimodel_flip.jsonl
LOG=results/log_localmm.txt
TARGET=2778   # 463 items x 2 protocols x 3 vectors per model

count(){ python3 -c "import json;print(sum(1 for l in open('$OUT') if l.strip() and json.loads(l).get('model')=='$1'))" 2>/dev/null || echo 0; }

run_model(){
  local M="$1" fails=0
  while true; do
    local n=$(count "$M")
    [ "$n" -ge "$TARGET" ] && { echo "=== $M COMPLETE ($n) ===" >>"$LOG"; break; }
    echo "=== $M start rows=$n $(date +%H:%M) ===" >>"$LOG"
    local t0=$(date +%s)
    MODEL="$M" OUT="$OUT" uv run python -u scripts/38_multimodel_run.py >>"$LOG" 2>&1 &
    local PID=$! last=$n stall=0
    while kill -0 $PID 2>/dev/null; do
      sleep 60
      local cur=$(count "$M")
      # 45-min startup grace: uncached vector build on a 4B (rimsky reads 500 items x2 on MPS) can
      # take ~35min and writes no rows -> would false-stall-kill mid-build. Cached builds start fast.
      [ $(( $(date +%s) - t0 )) -lt 2700 ] && { last=$cur; continue; }
      if [ "$cur" -le "$last" ]; then stall=$((stall+1)); else stall=0; fi
      last=$cur
      [ "$stall" -ge 8 ] && { echo "STALL kill $M @ $cur" >>"$LOG"; kill -9 $PID 2>/dev/null; break; }
    done
    local dt=$(( $(date +%s) - t0 )) after=$(count "$M")
    # crash-loop guard: exited fast with no progress -> likely OOM/load error
    if [ "$after" -le "$n" ] && [ "$dt" -lt 180 ]; then
      fails=$((fails+1)); echo "!! $M fast-exit no-progress ($fails) dt=${dt}s" >>"$LOG"
      [ "$fails" -ge 3 ] && { echo "=== $M BAILED (likely OOM/load fail) ===" >>"$LOG"; break; }
    else fails=0; fi
    sleep 20
  done
}

echo "===== LOCAL MM RUN $(date) =====" >>"$LOG"
run_model "unsloth/gemma-3-1b-it"
run_model "unsloth/gemma-3-4b-it"
echo "===== ALL DONE $(date) =====" >>"$LOG"
