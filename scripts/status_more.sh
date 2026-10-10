#!/usr/bin/env bash
# Progress:   bash /workspace/thesis_experiments/status_more.sh
P=/workspace/thesis_experiments; S=$P/src
date; pgrep -f run_more.sh >/dev/null && echo "STATUS: RUNNING" || echo "STATUS: NOT RUNNING"
grep "^====" /workspace/more.log 2>/dev/null | tail -6
for st in kgrand eps splits; do [ -f $P/.done_more_$st ] && echo "  done: $st" || echo "  todo: $st"; done
printf "  results: kgrand %s/10, eps %s/15, splits %s/30\n" $(ls $S/results_kgrand/fl_dp_seed*.json 2>/dev/null | wc -l) $(ls $S/results_eps*/fl_dp_seed*.json 2>/dev/null | wc -l) $(ls $S/results_split*/*_seed*.json 2>/dev/null | grep -v -E "dp_log" | wc -l)
echo "NOW RUNNING:"; ps -eo args | grep -E "python .*(--seed|data.py)" | grep -v grep | cut -c1-110 | head -2
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
