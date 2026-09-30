#!/usr/bin/env bash
# Validation-only tuning of the private model and lower-noise final runs. See RUN_GUIDE.md. Resumable.
#   nohup bash run_tune.sh > /workspace/tune.log 2>&1 &
cd "$(dirname "$0")/src"
[ -f /root/prepared/meta.json ] || { echo "prepare the data first (run_all.sh does it)"; exit 1; }
python tune_dp.py --stage all --prep /root/prepared --out results_tuned
