#!/usr/bin/env bash
# Four further experiments (about 21 h; about $5.50), in this order, each resumable:
#   userdp    tenant-level DP-FedAvg (server clips whole tenant updates and adds noise), clip 0.5 and 5, seeds 0-2
#   shuffled  shuffled-ontology control (CSO+ vectors exchanged between ports), DP-FL, seeds 0-9
#   scaling   50 and 100 tenants: DP-FL, FedAvg without DP, local random forests, seeds 0-2
#   replace   model-replacement attack (boosted updates, no DP) by 1 or 3 tenants vs FedAvg / FedMedian / norm bounding, seeds 0-2
#   cd /workspace && unzip -o -q extra3_package.zip -d thesis_experiments 2>/dev/null; nohup bash thesis_experiments/run_extra3.sh > /workspace/extra3.log 2>&1 &
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
CSV=/root/data/MachineLearningCVE; KG=../ontology/cso_plus_node2vec.npz; cd $S
[ -f /root/prepared/meta.json ] || python data.py --csv_dir $CSV --kg $KG --out_dir /root/prepared --n_tenants 10 --partition iid || { say "ERROR: prep"; exit 1; }
say "step 3/4: smoke test (tenant-level DP, model replacement with norm bounding, shuffled ontology, 10 small tenants)"
if [ ! -f $P/.extra3_smoke_ok ]; then
  SM=/root/smoke_x3; rm -rf $SM; mkdir -p $SM; F="--sigma 1.0 --rounds 2 --local_epochs 1 --fraction 0.6 --eval_every 1 --gpus_per_client 1.0"
  python make_smoke_data.py $SM/csv >/dev/null && \
  python data.py --csv_dir $SM/csv --kg $SM/csv/SMOKE_ONLY_kg_embeddings.npz --out_dir $SM/prep --n_tenants 5 --block 10 >/dev/null && \
  python make_shuffled_kg.py $SM/csv/SMOKE_ONLY_kg_embeddings.npz $SM/shuf.npz && \
  python data.py --csv_dir $SM/csv --kg $SM/shuf.npz --out_dir $SM/prep_shuf --n_tenants 5 --block 10 >/dev/null && \
  python data.py --csv_dir $SM/csv --kg $SM/csv/SMOKE_ONLY_kg_embeddings.npz --out_dir $SM/prep10 --n_tenants 10 --block 10 >/dev/null && \
  python fl_flower.py --prep $SM/prep --out $SM/out --dp off --clients 5 $F --user_dp --user_clip 0.5 && \
  python fl_flower.py --prep $SM/prep --out $SM/out --dp on --clients 5 $F --malicious 1 --attack replace --strategy normclip && \
  python fl_flower.py --prep $SM/prep_shuf --out $SM/out_shuf --dp on --clients 5 $F && \
  python fl_flower.py --prep $SM/prep10 --out $SM/out10 --dp on --clients 10 $F && \
  python local_rf.py --prep $SM/prep10 --out $SM/out10 --seed 0 --trees 10 \
    && [ -f $SM/out/fl_nodp_udp0.5_seed0.json ] && [ -f $SM/out/fl_dp_m1_replace_normclip_seed0.json ] && [ -f $SM/out_shuf/fl_dp_seed0.json ] && [ -f $SM/out10/local_rf_seed0.json ] \
    && touch $P/.extra3_smoke_ok && say "SMOKE TEST PASSED" || { say "ERROR: smoke test failed - nothing else was run"; exit 1; }
fi
say "step 4/4: experiments"
FL="--clients 10 --rounds 50 --local_epochs 5 --fraction 0.3"
retry() { local out=$1; shift; for attempt in 1 2; do run "$out" "$@" && return 0; ray stop --force >/dev/null 2>&1; sleep 20; done; return 1; }
stage_userdp() {
  for c in 0.5 5; do for s in 0 1 2; do
    retry results_userdp/fl_nodp_udp${c}_seed$s.json fl_flower.py --prep /root/prepared --out results_userdp --dp off $FL --seed $s --user_dp --user_sigma 1.74 --user_clip $c --gpus_per_client 0.25 || return 1
  done; done
}
stage_shuffled() {
  [ -f ../ontology/shuffled_kg.npz ] || python make_shuffled_kg.py $KG ../ontology/shuffled_kg.npz || return 1
  [ -f /root/prepared_skg/meta.json ] || python data.py --csv_dir $CSV --kg ../ontology/shuffled_kg.npz --out_dir /root/prepared_skg --n_tenants 10 --partition iid || return 1
  local SIG; SIG=$(python choose_sigma.py --prep /root/prepared_skg --rounds 50 --fraction 0.3 --local_epochs 5 --target_eps 8.0 --quiet); echo "shuffled sigma: $SIG"
  for s in 0 1 2 3 4 5 6 7 8 9; do retry results_kgshuf/fl_dp_seed$s.json fl_flower.py --prep /root/prepared_skg --out results_kgshuf --dp on --sigma $SIG $FL --seed $s --gpus_per_client 0.5 || return 1; done
}
stage_scaling() {
  for n in 50 100; do
    local PR=/root/prepared_t$n O=results_t$n
    [ -f $PR/meta.json ] || python data.py --csv_dir $CSV --kg $KG --out_dir $PR --n_tenants $n --partition iid || return 1
    local SIG; SIG=$(python choose_sigma.py --prep $PR --rounds 50 --fraction 0.3 --local_epochs 5 --target_eps 8.0 --quiet); echo "tenants $n sigma: $SIG"
    for s in 0 1 2; do
      run $O/local_rf_seed$s.json local_rf.py --prep $PR --out $O --seed $s || return 1
      retry $O/fl_nodp_seed$s.json fl_flower.py --prep $PR --out $O --dp off --clients $n --rounds 50 --local_epochs 5 --fraction 0.3 --seed $s --gpus_per_client 0.25 || return 1
      retry $O/fl_dp_seed$s.json fl_flower.py --prep $PR --out $O --dp on --sigma $SIG --clients $n --rounds 50 --local_epochs 5 --fraction 0.3 --seed $s --gpus_per_client 1.0 || return 1
    done
  done
}
stage_replace() {
  local st
  for s in 0 1 2; do
    retry results_replace/fl_dp_normclip_seed$s.json fl_flower.py --prep /root/prepared --out results_replace --dp on --sigma 1.455 $FL --seed $s --strategy normclip --gpus_per_client 0.5 || return 1
    for m in 1 3; do for st in fedavg median normclip; do
      case $st in fedavg) tag="fl_dp_m${m}_replace";; median) tag="fl_dp_m${m}_median_replace";; normclip) tag="fl_dp_m${m}_replace_normclip";; esac   # same order as TAG in fl_flower.py
      retry results_replace/${tag}_seed$s.json fl_flower.py --prep /root/prepared --out results_replace --dp on --sigma 1.455 $FL --seed $s --malicious $m --attack replace --boost 10 --strategy $st --gpus_per_client 0.5 || return 1
    done; done
  done
}
FAILED=""
for st in userdp shuffled scaling replace; do
  if [ -f $P/.done_x3_$st ]; then say "stage $st already finished, skipping"; continue; fi
  say "stage $st started"
  if stage_$st; then touch $P/.done_x3_$st; say "stage $st finished"; else FAILED="$FAILED $st"; say "stage $st FAILED"; fi
  python make_extra3_tables.py >/dev/null 2>&1
  ( zip -q -r $W/extra3_results.zip results_userdp results_kgshuf results_t50 results_t100 results_replace extra3_results.md -x '*.pt' 2>/dev/null ) || true
done
if [ -z "$FAILED" ]; then say "ALL FINISHED"; else say "FINISHED WITH FAILED STAGES:$FAILED"; fi
say "download /workspace/extra3_results.zip and /workspace/extra3.log"
if [ "$AUTO_STOP" = "1" ] && command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then say "stopping the pod in 10 minutes"; sleep 600; runpodctl stop pod "$RUNPOD_POD_ID"; fi
