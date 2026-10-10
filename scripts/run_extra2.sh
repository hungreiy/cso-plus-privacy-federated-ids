#!/usr/bin/env bash
# Architecture baselines (MLP, Transformer under the same DP-FL) and the stronger membership-inference attack (offline LiRA).
# About 5 h. Needs the main-study weights results/central_seed{0,1,2}.pt and results/fl_dp_seed{0,1,2}.pt on the volume.
#   cd /workspace && unzip -o -q extra2_package.zip -d thesis_experiments 2>/dev/null; nohup bash thesis_experiments/run_extra2.sh > /workspace/extra2.log 2>&1 &
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
for s in 0 1 2; do for m in central fl_dp; do [ -f results/${m}_seed$s.pt ] || say "WARNING: results/${m}_seed$s.pt not found - LiRA will skip that target"; done; done
say "step 3/4: smoke test (MLP and Transformer under DP-FL, LiRA with 2 shadows, all on synthetic data)"
if [ ! -f $P/.extra2_smoke_ok ]; then
  SM=/root/smoke_x2; rm -rf $SM; mkdir -p $SM
  python make_smoke_data.py $SM/csv >/dev/null && \
  python data.py --csv_dir $SM/csv --kg $SM/csv/SMOKE_ONLY_kg_embeddings.npz --out_dir $SM/prep --n_tenants 5 --block 10 >/dev/null && \
  python fl_flower.py --prep $SM/prep --out $SM/out --dp on --sigma 1.0 --clients 5 --rounds 2 --local_epochs 1 --fraction 0.6 --eval_every 1 --arch mlp --gpus_per_client 1.0 && \
  python fl_flower.py --prep $SM/prep --out $SM/out --dp on --sigma 1.0 --clients 5 --rounds 2 --local_epochs 1 --fraction 0.6 --eval_every 1 --arch transformer --gpus_per_client 1.0 && \
  python fl_flower.py --prep $SM/prep --out $SM/out --dp on --sigma 1.0 --clients 5 --rounds 2 --local_epochs 1 --fraction 0.6 --eval_every 1 --gpus_per_client 1.0 && \
  python train_central.py --prep $SM/prep --out $SM/out --seed 0 --epochs 2 && \
  python mia_lira.py --prep $SM/prep --targets central:$SM/out/central_seed0.pt:0 fl_dp:$SM/out/fl_dp_seed0.pt:1 --shadows 2 --epochs 1 --n 40 --out $SM/out/lira.json \
    && [ -f $SM/out/fl_dp_mlp_seed0.json ] && [ -f $SM/out/fl_dp_transformer_seed0.json ] && [ -f $SM/out/lira.json ] && touch $P/.extra2_smoke_ok && say "SMOKE TEST PASSED" || { say "ERROR: smoke test failed - nothing else was run"; exit 1; }
fi
say "step 4/4: experiments"
FL="--dp on --sigma 1.455 --prep /root/prepared --out results_arch --clients 10 --rounds 50 --local_epochs 5 --fraction 0.3 --gpus_per_client 1.0"
FAILED=""
for s in 0 1 2; do
  for ar in mlp transformer; do
    say "architecture $ar, seed $s"
    for attempt in 1 2; do run results_arch/fl_dp_${ar}_seed$s.json fl_flower.py $FL --seed $s --arch $ar && break; ray stop --force >/dev/null 2>&1; sleep 20; done
    [ -f results_arch/fl_dp_${ar}_seed$s.json ] || FAILED="$FAILED ${ar}_$s"
  done
  say "LiRA, target seed $s"
  run results_mia2/lira_seed$s.json mia_lira.py --prep /root/prepared --targets central:results/central_seed$s.pt:0 fl_dp:results/fl_dp_seed$s.pt:1 --shadows 8 --epochs 10 --n 2000 --seed $s --out results_mia2/lira_seed$s.json || FAILED="$FAILED lira_$s"
  ( zip -q -r $W/extra2_results.zip results_arch results_mia2 -x '*.pt' ) || true
done
python make_extra2_tables.py || true
( zip -q -r $W/extra2_results.zip results_arch results_mia2 extra2_results.md -x '*.pt' ) || true
if [ -z "$FAILED" ]; then say "ALL FINISHED"; else say "FINISHED WITH FAILURES:$FAILED"; fi
say "download /workspace/extra2_results.zip and /workspace/extra2.log"
if [ "$AUTO_STOP" = "1" ] && command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then say "stopping the pod in 10 minutes"; sleep 600; runpodctl stop pod "$RUNPOD_POD_ID"; fi
