#!/usr/bin/env bash
# Label-scarce tenants: only 0.5 %, 2 % or 10 % of each tenant's windows are labelled (about 6.5 h).
#   local random forests (labelled windows only) vs DP-FL with KG (classifier on labelled windows, reconstruction also on unlabelled ones)
#   cd /workspace && unzip -o -q label_package.zip -d thesis_experiments 2>/dev/null; nohup bash thesis_experiments/run_label.sh > /workspace/label.log 2>&1 &
set -uo pipefail
W=/workspace; P=$W/thesis_experiments; S=$P/src; AUTO_STOP=${AUTO_STOP:-1}
export UV_CACHE_DIR=/root/.cache/uv PIP_CACHE_DIR=/root/.cache/pip XDG_CACHE_HOME=/root/.cache UV_LINK_MODE=copy
say() { echo "==== [$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
run() { local out=$1; shift; if [ -f "$out" ]; then echo "skip $out"; else python "$@" || return 1; fi; }
say "step 1/4: Python environment"
if [ ! -x /root/venv311/bin/python ]; then pip install -q uv && uv venv --python 3.11 /root/venv311 || exit 1; fi
source /root/venv311/bin/activate; cd $P && uv pip install -q -r requirements.txt || { say "ERROR: packages"; exit 1; }
say "step 2/4: data"
if [ "$(ls /root/data/MachineLearningCVE/*.csv 2>/dev/null | wc -l)" -lt 8 ]; then
  mkdir -p /root/data && cd /root/data
  for t in 1 2 3; do curl -L --retry 5 -o MachineLearningCSV.zip "https://huggingface.co/datasets/Mireu-Lab/CIC-IDS/resolve/main/ZIP/MachineLearningCSV.zip" && break; sleep 60; done
  [ "$(sha256sum MachineLearningCSV.zip | cut -c1-32)" = "c3f26274b36c837ccf28ffd2dbf45829" ] || { say "ERROR: checksum"; exit 1; }
  unzip -o -q MachineLearningCSV.zip -d /root/data && rm -f MachineLearningCSV.zip
fi
cd $S
[ -f /root/prepared/meta.json ] || python data.py --csv_dir /root/data/MachineLearningCVE --kg ../ontology/cso_plus_node2vec.npz --out_dir /root/prepared --n_tenants 10 --partition iid || { say "ERROR: prep"; exit 1; }
say "step 3/4: smoke test (label fraction 0.2, 2 rounds on synthetic data)"
if [ ! -f $P/.label_smoke_ok ]; then
  rm -rf /root/smoke_lab_csv /root/prep_smoke_lab /root/smoke_lab_out
  python make_smoke_data.py /root/smoke_lab_csv >/dev/null && \
  python data.py --csv_dir /root/smoke_lab_csv --kg /root/smoke_lab_csv/SMOKE_ONLY_kg_embeddings.npz --out_dir /root/prep_smoke_lab --n_tenants 5 --block 10 >/dev/null && \
  python fl_flower.py --prep /root/prep_smoke_lab --out /root/smoke_lab_out --dp on --sigma 1.0 --clients 5 --rounds 2 --local_epochs 1 --fraction 0.6 --eval_every 1 --label_frac 0.2 --gpus_per_client 1.0 && \
  python local_rf.py --prep /root/prep_smoke_lab --out /root/smoke_lab_out --seed 0 --trees 10 --label_frac 0.2 \
    && [ -f /root/smoke_lab_out/fl_dp_lf0.2_seed0.json ] && [ -f /root/smoke_lab_out/local_rf_lf0.2_seed0.json ] && touch $P/.label_smoke_ok && say "SMOKE TEST PASSED" || { say "ERROR: smoke test failed - nothing else was run"; exit 1; }
fi
say "step 4/4: experiments"
FL="--dp on --sigma 1.455 --prep /root/prepared --out results_label --clients 10 --rounds 50 --local_epochs 5 --fraction 0.3 --gpus_per_client 1.0"
FAILED=""
for f in 0.005 0.02 0.1; do
  for s in 0 1 2; do
    say "label fraction $f, seed $s"
    run results_label/local_rf_lf${f}_seed$s.json local_rf.py --prep /root/prepared --out results_label --seed $s --label_frac $f || FAILED="$FAILED rf_lf${f}_$s"
    for attempt in 1 2; do run results_label/fl_dp_lf${f}_seed$s.json fl_flower.py $FL --seed $s --label_frac $f && break; ray stop --force >/dev/null 2>&1; sleep 20; done
    [ -f results_label/fl_dp_lf${f}_seed$s.json ] || FAILED="$FAILED fl_lf${f}_$s"
    ( zip -q -r $W/label_results.zip results_label -x '*.pt' ) || true
  done
done
python make_label_tables.py || true
( zip -q -r $W/label_results.zip results_label label_results.md -x '*.pt' ) || true
if [ -z "$FAILED" ]; then say "ALL FINISHED"; else say "FINISHED WITH FAILURES:$FAILED"; fi
say "download /workspace/label_results.zip and /workspace/label.log"
if [ "$AUTO_STOP" = "1" ] && command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then say "stopping the pod in 10 minutes"; sleep 600; runpodctl stop pod "$RUNPOD_POD_ID"; fi
