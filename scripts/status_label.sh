#!/usr/bin/env bash
S=/workspace/thesis_experiments/src; date
pgrep -f run_label.sh >/dev/null && echo "STATUS: RUNNING" || echo "STATUS: NOT RUNNING"
grep "^====" /workspace/label.log 2>/dev/null | tail -4
echo "results: DP-FL $(ls $S/results_label/fl_dp_lf*_seed*.json 2>/dev/null | wc -l)/9, local RF $(ls $S/results_label/local_rf_lf*_seed*.json 2>/dev/null | wc -l)/9"
ps -eo args | grep -E "python .*--seed" | grep -v grep | cut -c1-110 | head -1
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
