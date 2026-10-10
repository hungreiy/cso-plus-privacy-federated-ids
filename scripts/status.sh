#!/usr/bin/env bash
# Progress of run_everything.sh:   bash /workspace/thesis_experiments/status.sh
L=/workspace/everything.log
P=/workspace/thesis_experiments
if pgrep -f run_everything.sh >/dev/null; then echo "STATUS: RUNNING"; else echo "STATUS: NOT RUNNING"; fi
echo; echo "--- milestones ---"; grep "^====" $L 2>/dev/null | tail -12
echo; echo "--- stages finished ---"; for st in seeds baselines noniid ds2018 cross window; do [ -f $P/.done_$st ] && echo "  done: $st" || echo "  todo: $st"; done
echo; echo "--- result files so far ---"
for d in results results_2018 results_noniid results_cross results_w5 results_w30; do
  n=$(ls $P/src/$d/*.json 2>/dev/null | wc -l); printf "  %-16s %3s files\n" $d $n; done
echo; echo "--- last lines of the log ---"; tail -3 $L 2>/dev/null
echo; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
