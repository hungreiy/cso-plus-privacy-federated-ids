#!/usr/bin/env bash
# ONE command for all additional thesis experiments (about 32 h on an RTX 4000 Ada).
# Start it once and close the browser; it keeps running on the pod:
#
#   cd /workspace && unzip -o -q pod_package.zip && nohup bash thesis_experiments/run_everything.sh > /workspace/everything.log 2>&1 &
#
# Safe to start again after a crash or a pod restart: finished runs are skipped.
# Progress:  bash /workspace/thesis_experiments/status.sh
# When everything has finished, the results are in /workspace/extra_results.zip and the pod is stopped
# automatically (set AUTO_STOP=0 before the command to keep it running).

set -uo pipefail
W=/workspace
P=$W/thesis_experiments
AUTO_STOP=${AUTO_STOP:-1}
STAGES="seeds baselines noniid ds2018 cross window"
say() { echo "==== [$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

backup() {   # results only, without model weights
  ( cd $P/src && zip -q -r $W/extra_results.zip results results_2018 results_noniid results_cross results_w5 results_w30 \
      results_extra.md -x '*.pt' 2>/dev/null ) || true
  say "backup written to $W/extra_results.zip"
}

[ -d $P/src ] || { say "ERROR: $P/src not found. Upload pod_package.zip to /workspace and unzip it first."; exit 1; }
[ -d $P/src/results ] || say "WARNING: $P/src/results not found; the main study (seeds 0-4) will be recomputed by the 'seeds' stage only for seeds 5-9."

# ---------- 1. Python environment (the container disk /root is wiped when a pod restarts) ----------
say "step 1/5: Python environment"
if [ ! -x /root/venv311/bin/python ]; then
  pip install -q uv && uv venv --python 3.11 /root/venv311 || { say "ERROR: could not create the virtual environment"; exit 1; }
fi
source /root/venv311/bin/activate
cd $P && uv pip install -q --link-mode=copy -r requirements.txt && uv pip install -q xgboost awscli \
  || { say "ERROR: package installation failed"; exit 1; }
python -c "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.is_available() and torch.cuda.get_device_name(0))"

# ---------- 2. CIC-IDS2017 (needed again after a pod restart) ----------
say "step 2/5: CIC-IDS2017 data"
if [ "$(ls /root/data/MachineLearningCVE/*.csv 2>/dev/null | wc -l)" -lt 8 ]; then
  mkdir -p /root/data && cd /root/data
  for try in 1 2 3; do
    curl -L --retry 5 -o MachineLearningCSV.zip "https://huggingface.co/datasets/Mireu-Lab/CIC-IDS/resolve/main/ZIP/MachineLearningCSV.zip" && break
    say "download attempt $try failed, retrying in 60 s"; sleep 60
  done
  [ "$(sha256sum MachineLearningCSV.zip | cut -c1-32)" = "c3f26274b36c837ccf28ffd2dbf45829" ] \
    || { say "ERROR: CIC-IDS2017 checksum mismatch"; exit 1; }
  unzip -o -q MachineLearningCSV.zip -d /root/data && rm -f MachineLearningCSV.zip
fi
say "CIC-IDS2017 files: $(ls /root/data/MachineLearningCVE/*.csv | wc -l) (expected 8)"

# ---------- 3. disk space for CSE-CIC-IDS2018 (about 7 GB download) ----------
say "step 3/5: disk space"
FREE=$(df -BG --output=avail /root | tail -1 | tr -dc '0-9')
say "free space on /root: ${FREE} GB"
if [ ! -f /root/data18/.done ] && [ "$FREE" -lt 12 ]; then
  say "WARNING: less than 20 GB free; skipping the ds2018 and cross stages (increase the container disk to run them)"
  STAGES="seeds baselines noniid window"
fi

# ---------- 4. smoke test on fake data (about 5-8 minutes) ----------
say "step 4/5: smoke test"
cd $P
if [ -f $P/.smoke_passed ]; then
  say "smoke test already passed earlier, skipping"
else
  bash run_smoke_extra.sh | tee /workspace/smoke_extra.log
  grep -q "EXTRA SMOKE TEST PASSED" /workspace/smoke_extra.log \
    || { say "ERROR: smoke test failed - see /workspace/smoke_extra.log. Nothing else was run."; exit 1; }
  touch $P/.smoke_passed
fi

# ---------- 5. the experiments, one stage at a time; a failed stage is retried once, then skipped ----------
say "step 5/5: experiments, stages: $STAGES"
FAILED=""
for st in $STAGES; do
  if [ -f $P/.done_$st ]; then say "stage $st already finished, skipping"; continue; fi
  ok=0
  for attempt in 1 2; do
    say "stage $st: attempt $attempt started"
    if bash $P/run_extra.sh $st; then ok=1; break; fi
    say "stage $st: attempt $attempt FAILED"; sleep 30
  done
  if [ $ok -eq 1 ]; then touch $P/.done_$st; say "stage $st finished"; else FAILED="$FAILED $st"; fi
  backup
done

cd $P/src && python make_extra_tables.py . >/dev/null 2>&1 || true
backup
if [ -z "$FAILED" ]; then say "ALL STAGES FINISHED"; else say "FINISHED WITH FAILED STAGES:$FAILED (see this log)"; fi
say "download /workspace/extra_results.zip and /workspace/everything.log"

if [ "$AUTO_STOP" = "1" ] && command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
  say "stopping the pod in 10 minutes to save credit (/workspace is kept)"; sleep 600
  runpodctl stop pod "$RUNPOD_POD_ID"
fi
