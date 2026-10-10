#!/usr/bin/env bash
# Three follow-up experiments (about 23 h on an RTX 2000 Ada; about $6):
#   kgrand  random-embedding control for the KG result: DP-FL with random port vectors, seeds 0-9
#   eps     privacy-utility curve: DP-FL at target epsilon 1, 2, 4 (8 = main study), seeds 0-4
#   splits  two more block-wise data splits (data seeds 1, 2): rf, central, fl_nodp, fl_dp, fl_dp_nokg, seeds 0-2
# Start once (safe to start again; finished runs are skipped):
#   cd /workspace && unzip -o -q more_package.zip -d thesis_experiments 2>/dev/null; nohup bash thesis_experiments/run_more.sh > /workspace/more.log 2>&1 &
set -uo pipefail
W=/workspace; P=$W/thesis_experiments; S=$P/src
AUTO_STOP=${AUTO_STOP:-1}
export UV_CACHE_DIR=/root/.cache/uv PIP_CACHE_DIR=/root/.cache/pip XDG_CACHE_HOME=/root/.cache UV_LINK_MODE=copy
say() { echo "==== [$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
backup() { ( cd $S && zip -q -r $W/more_results.zip results_kgrand results_eps1 results_eps2 results_eps4 results_split1 results_split2 more_results.md -x '*.pt' 2>/dev/null ) || true; say "backup written to $W/more_results.zip"; }
run() { local out=$1; shift; if [ -f "$out" ]; then echo "skip $out"; else python "$@" || return 1; fi; }

say "step 1/4: Python environment"
if [ ! -x /root/venv311/bin/python ]; then pip install -q uv && uv venv --python 3.11 /root/venv311 || { say "ERROR: venv"; exit 1; }; fi
source /root/venv311/bin/activate
cd $P && uv pip install -q -r requirements.txt && uv pip install -q xgboost || { say "ERROR: package installation failed"; exit 1; }
python -c "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.is_available() and torch.cuda.get_device_name(0))"

say "step 2/4: CIC-IDS2017 data"
if [ "$(ls /root/data/MachineLearningCVE/*.csv 2>/dev/null | wc -l)" -lt 8 ]; then
  mkdir -p /root/data && cd /root/data
  for try in 1 2 3; do curl -L --retry 5 -o MachineLearningCSV.zip "https://huggingface.co/datasets/Mireu-Lab/CIC-IDS/resolve/main/ZIP/MachineLearningCSV.zip" && break; sleep 60; done
  [ "$(sha256sum MachineLearningCSV.zip | cut -c1-32)" = "c3f26274b36c837ccf28ffd2dbf45829" ] || { say "ERROR: checksum"; exit 1; }
  unzip -o -q MachineLearningCSV.zip -d /root/data && rm -f MachineLearningCSV.zip
fi
say "CIC-IDS2017 files: $(ls /root/data/MachineLearningCVE/*.csv | wc -l)"

CSV=/root/data/MachineLearningCVE; KG=../ontology/cso_plus_node2vec.npz
FL="--clients 10 --rounds 50 --local_epochs 5 --fraction 0.3 --gpus_per_client 1.0"
cd $S
say "step 3/4: main data split (same as the main study: data seed 0)"
[ -f /root/prepared/meta.json ] || python data.py --csv_dir $CSV --kg $KG --out_dir /root/prepared --n_tenants 10 --partition iid || { say "ERROR: prep"; exit 1; }

stage_kgrand() {
  [ -f ../ontology/random_kg.npz ] || python make_random_kg.py $KG ../ontology/random_kg.npz || return 1
  [ -f /root/prepared_rkg/meta.json ] || python data.py --csv_dir $CSV --kg ../ontology/random_kg.npz --out_dir /root/prepared_rkg --n_tenants 10 --partition iid || return 1
  local SIG; SIG=$(python choose_sigma.py --prep /root/prepared_rkg --rounds 50 --fraction 0.3 --local_epochs 5 --target_eps 8.0 --quiet); echo "kgrand sigma: $SIG"
  for s in 0 1 2 3 4 5 6 7 8 9; do run results_kgrand/fl_dp_seed$s.json fl_flower.py --seed $s --dp on --sigma $SIG --prep /root/prepared_rkg --out results_kgrand $FL || return 1; done
}
stage_eps() {
  for e in 1 2 4; do
    local SIG; SIG=$(python choose_sigma.py --prep /root/prepared --rounds 50 --fraction 0.3 --local_epochs 5 --target_eps $e --quiet); echo "eps $e sigma: $SIG"
    for s in 0 1 2 3 4; do run results_eps$e/fl_dp_seed$s.json fl_flower.py --seed $s --dp on --sigma $SIG --prep /root/prepared --out results_eps$e $FL || return 1; done
  done
}
stage_splits() {
  for sp in 1 2; do
    local PR=/root/prepared_split$sp O=results_split$sp
    [ -f $PR/meta.json ] || python data.py --csv_dir $CSV --kg $KG --out_dir $PR --n_tenants 10 --partition iid --seed $sp || return 1
    local SIG; SIG=$(python choose_sigma.py --prep $PR --rounds 50 --fraction 0.3 --local_epochs 5 --target_eps 8.0 --quiet); echo "split $sp sigma: $SIG"
    for s in 0 1 2; do
      run $O/rf_seed$s.json train_rf.py --prep $PR --out $O --seed $s || return 1
      run $O/central_seed$s.json train_central.py --prep $PR --out $O --seed $s || return 1
      run $O/fl_nodp_seed$s.json fl_flower.py --seed $s --dp off --prep $PR --out $O $FL || return 1
      run $O/fl_dp_seed$s.json fl_flower.py --seed $s --dp on --sigma $SIG --prep $PR --out $O $FL || return 1
      run $O/fl_dp_nokg_seed$s.json fl_flower.py --seed $s --dp on --sigma $SIG --prep $PR --out $O $FL --no_kg || return 1
    done
  done
}

say "step 4/4: experiments"
FAILED=""
for st in kgrand eps splits; do
  if [ -f $P/.done_more_$st ]; then say "stage $st already finished, skipping"; continue; fi
  ok=0
  for attempt in 1 2; do
    say "stage $st: attempt $attempt started"
    if stage_$st; then ok=1; break; fi
    say "stage $st: attempt $attempt FAILED"; ray stop --force >/dev/null 2>&1; sleep 30
  done
  if [ $ok -eq 1 ]; then touch $P/.done_more_$st; say "stage $st finished"; else FAILED="$FAILED $st"; fi
  python make_more_tables.py >/dev/null 2>&1; backup
done
if [ -z "$FAILED" ]; then say "ALL STAGES FINISHED"; else say "FINISHED WITH FAILED STAGES:$FAILED"; fi
say "download /workspace/more_results.zip and /workspace/more.log"
if [ "$AUTO_STOP" = "1" ] && command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then say "stopping the pod in 10 minutes"; sleep 600; runpodctl stop pod "$RUNPOD_POD_ID"; fi
