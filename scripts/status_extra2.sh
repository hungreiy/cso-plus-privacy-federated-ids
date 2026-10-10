#!/usr/bin/env bash
S=/workspace/thesis_experiments/src; date
pgrep -f run_extra2.sh >/dev/null && echo "STATUS: RUNNING" || echo "STATUS: NOT RUNNING"
grep "^====" /workspace/extra2.log 2>/dev/null | tail -4
echo "results: architectures $(ls $S/results_arch/fl_dp_*_seed*.json 2>/dev/null | wc -l)/6, LiRA $(ls $S/results_mia2/lira_seed*.json 2>/dev/null | wc -l)/3"
ps -eo args | grep -E "python .*(--seed|mia_lira)" | grep -v grep | cut -c1-110 | head -1
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
