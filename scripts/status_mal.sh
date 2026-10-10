#!/usr/bin/env bash
S=/workspace/thesis_experiments/src; date
pgrep -f run_mal.sh >/dev/null && echo "STATUS: RUNNING" || echo "STATUS: NOT RUNNING"
grep "^====" /workspace/mal.log 2>/dev/null | tail -4
echo "results: $(ls $S/results_mal/*_seed*.json 2>/dev/null | grep -v dp_log | wc -l)/15"
ps -eo args | grep -E "python .*--seed" | grep -v grep | cut -c1-110 | head -1
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
