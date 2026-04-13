#!/bin/bash
# run_forever.sh — Restart STK training if it crashes (segfault)
# The checkpoint system ensures no progress is lost.
cd /home/clawadmin/neat-racer

while true; do
    echo "[RUNNER] Starting STK SAC training..."
    python3 training/stk_sac.py 2>&1 | tee -a /tmp/stk_stream.log
    EXIT_CODE=$?
    echo "[RUNNER] Process exited with code $EXIT_CODE — restarting in 5s..."
    sleep 5
done
