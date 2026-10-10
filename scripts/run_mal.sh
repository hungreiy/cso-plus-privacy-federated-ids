#!/usr/bin/env bash
# Malicious tenants (label flipping) and robust aggregation (FedMedian), DP-FL with KG, sigma 1.455, seeds 0-2 (about 10 h):
#   malicious tenants 0 / 1 / 3 of 10  x  aggregation FedAvg / FedMedian   (0 + FedAvg = main study, reused)
#   cd /workspace && unzip -o -q mal_package.zip -d thesis_experiments 2>/dev/null; nohup bash thesis_experiments/run_mal.sh > /workspace/mal.log 2>&1 &
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
say "step 3/4: smoke test (1 malicious tenant, FedMedian, 2 rounds on synthetic data)"
if [ ! -f $P/.mal_smoke_ok ]; then
  rm -rf /root/smoke_mal_csv /root/prep_smoke_mal /root/smoke_mal_out
  python make_smoke_data.py /root/smoke_mal_csv >/dev/null && \
  python data.py --csv_dir /root/smoke_mal_csv --kg /root/smoke_mal_csv/SMOKE_ONLY_kg_embeddings.npz --out_dir /root/prep_smoke_mal --n_tenants 5 --block 10 >/dev/null && \
  python fl_flower.py --prep /root/prep_smoke_mal --out /root/smoke_mal_out --dp on --sigma 1.0 --clients 5 --rounds 2 --local_epochs 1 --fraction 0.6 --eval_every 1 --malicious 1 --strategy median --gpus_per_client 1.0 \
    && [ -f /root/smoke_mal_out/fl_dp_m1_median_seed0.json ] && touch $P/.mal_smoke_ok && say "SMOKE TEST PASSED" || { say "ERROR: smoke test failed - nothing else was run"; exit 1; }
fi
say "step 4/4: experiments"
FL="--dp on --sigma 1.455 --prep /root/prepared --out results_mal --clients 10 --rounds 50 --local_epochs 5 --fraction 0.3 --gpus_per_client 1.0"
FAILED=""
for s in 0 1 2; do
  for cfg in "0 median" "1 fedavg" "1 median" "3 fedavg" "3 median"; do
    set -- $cfg; m=$1; st=$2
    tag="fl_dp"; [ "$m" != "0" ] && tag="${tag}_m$m"; [ "$st" = "median" ] && tag="${tag}_median"
    say "seed $s: $m malicious, $st"
    for attempt in 1 2; do run results_mal/${tag}_seed$s.json fl_flower.py $FL --seed $s --malicious $m --strategy $st && break; ray stop --force >/dev/null 2>&1; sleep 20; done
    [ -f results_mal/${tag}_seed$s.json ] || FAILED="$FAILED ${tag}_seed$s"
    ( zip -q -r $W/mal_results.zip results_mal -x '*.pt' ) || true
  done
done
python make_mal_tables.py || true
( zip -q -r $W/mal_results.zip results_mal mal_results.md -x '*.pt' ) || true
if [ -z "$FAILED" ]; then say "ALL FINISHED"; else say "FINISHED WITH FAILURES:$FAILED"; fi
say "download /workspace/mal_results.zip and /workspace/mal.log"
if [ "$AUTO_STOP" = "1" ] && command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then say "stopping the pod in 10 minutes"; sleep 600; runpodctl stop pod "$RUNPOD_POD_ID"; fi
