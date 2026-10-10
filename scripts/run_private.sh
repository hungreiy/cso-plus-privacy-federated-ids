#!/usr/bin/env bash
# Private non-deep baselines (CPU only, about 30-60 min):
#   local_rf    one random forest per tenant on its own data only, seeds 0-4
#   dp_linear   federated logistic regression with the same DP-SGD (sigma 1.455) and accounting as the main model, seeds 0-4
#   cd /workspace && unzip -o -q private_package.zip -d thesis_experiments 2>/dev/null; nohup bash thesis_experiments/run_private.sh > /workspace/private.log 2>&1 &
set -uo pipefail
W=/workspace; P=$W/thesis_experiments; S=$P/src; AUTO_STOP=${AUTO_STOP:-1}
export UV_CACHE_DIR=/root/.cache/uv PIP_CACHE_DIR=/root/.cache/pip XDG_CACHE_HOME=/root/.cache UV_LINK_MODE=copy
say() { echo "==== [$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
run() { local out=$1; shift; if [ -f "$out" ]; then echo "skip $out"; else python "$@" || return 1; fi; }
say "step 1/3: Python environment"
if [ ! -x /root/venv311/bin/python ]; then pip install -q uv && uv venv --python 3.11 /root/venv311 || exit 1; fi
source /root/venv311/bin/activate; cd $P && uv pip install -q -r requirements.txt || { say "ERROR: packages"; exit 1; }
say "step 2/3: data"
if [ "$(ls /root/data/MachineLearningCVE/*.csv 2>/dev/null | wc -l)" -lt 8 ]; then
  mkdir -p /root/data && cd /root/data
  for t in 1 2 3; do curl -L --retry 5 -o MachineLearningCSV.zip "https://huggingface.co/datasets/Mireu-Lab/CIC-IDS/resolve/main/ZIP/MachineLearningCSV.zip" && break; sleep 60; done
  [ "$(sha256sum MachineLearningCSV.zip | cut -c1-32)" = "c3f26274b36c837ccf28ffd2dbf45829" ] || { say "ERROR: checksum"; exit 1; }
  unzip -o -q MachineLearningCSV.zip -d /root/data && rm -f MachineLearningCSV.zip
fi
cd $S
[ -f /root/prepared/meta.json ] || python data.py --csv_dir /root/data/MachineLearningCVE --kg ../ontology/cso_plus_node2vec.npz --out_dir /root/prepared --n_tenants 10 --partition iid || { say "ERROR: prep"; exit 1; }
say "step 3/3: experiments"
FAILED=""
for s in 0 1 2 3 4; do
  run results_private/local_rf_seed$s.json local_rf.py --prep /root/prepared --out results_private --seed $s || FAILED="$FAILED local_rf$s"
  run results_private/dp_linear_fl_seed$s.json dp_linear_fl.py --prep /root/prepared --out results_private --seed $s --sigma 1.455 || FAILED="$FAILED dp_linear$s"
  ( zip -q -r $W/private_results.zip results_private -x '*.pt' ) || true
done
python make_private_tables.py || true
( zip -q -r $W/private_results.zip results_private private_results.md ) || true
if [ -z "$FAILED" ]; then say "ALL FINISHED"; else say "FINISHED WITH FAILURES:$FAILED"; fi
say "download /workspace/private_results.zip and /workspace/private.log"
if [ "$AUTO_STOP" = "1" ] && command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then say "stopping the pod in 10 minutes"; sleep 600; runpodctl stop pod "$RUNPOD_POD_ID"; fi
