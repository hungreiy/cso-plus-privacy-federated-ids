#!/usr/bin/env bash
P=/workspace/thesis_experiments; S=$P/src; date
pgrep -f run_extra3.sh >/dev/null && echo "STATUS: RUNNING" || echo "STATUS: NOT RUNNING"
grep "^====" /workspace/extra3.log 2>/dev/null | tail -4
for st in userdp shuffled scaling replace; do [ -f $P/.done_x3_$st ] && echo "  done: $st" || echo "  todo: $st"; done
printf "  results: userdp %s/6, shuffled %s/10, scaling %s/18, replace %s/21\n" $(ls $S/results_userdp/*_seed*.json 2>/dev/null | wc -l) $(ls $S/results_kgshuf/fl_dp_seed*.json 2>/dev/null | wc -l) $(ls $S/results_t50/*_seed*.json $S/results_t100/*_seed*.json 2>/dev/null | wc -l) $(ls $S/results_replace/*_seed*.json 2>/dev/null | wc -l)
ps -eo args | grep -E "python .*--seed" | grep -v grep | cut -c1-110 | head -1
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
