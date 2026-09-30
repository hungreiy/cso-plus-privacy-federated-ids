#!/usr/bin/env bash
# Full Chapter 4 experiment. Resumable: finished runs are skipped.
#   nohup bash run_all.sh > /workspace/run_all.log 2>&1 &       then: tail -3 /workspace/run_all.log
# Run `bash run_smoke.sh` and `bash run_pilot.sh` first.
# Seeds: 0-4 by default (about 2 h per seed on an A5000-class GPU). Add more seeds later by editing SEEDS and re-running
# (finished runs are skipped). The Wilcoxon test needs at least 6 paired seeds; 8 or more are needed to reach p < 0.01.
set -e
cd "$(dirname "$0")/src"
CSV=/root/data/MachineLearningCVE
KG=../ontology/cso_plus_node2vec.npz
PREP=/root/prepared
TENANTS=10; ROUNDS=50; EPOCHS=5; FRACTION=0.3; TARGET_EPS=8.0
SEEDS="0 1 2 3 4"
[ -f $PREP/meta.json ] || python data.py --csv_dir $CSV --kg $KG --out_dir $PREP --n_tenants $TENANTS --partition iid
SIGMA=$(python choose_sigma.py --prep $PREP --rounds $ROUNDS --fraction $FRACTION --local_epochs $EPOCHS --target_eps $TARGET_EPS --quiet)
echo "noise multiplier chosen for target epsilon $TARGET_EPS: $SIGMA"
FL="--prep $PREP --clients $TENANTS --rounds $ROUNDS --local_epochs $EPOCHS --fraction $FRACTION"
run() { out=$1; shift; [ -f "results/$out" ] || python "$@"; }
for s in $SEEDS; do
  echo "=== seed $s ($(date))"
  run rf_seed$s.json             train_rf.py --prep $PREP --seed $s
  run central_seed$s.json        train_central.py --prep $PREP --seed $s
  run central_nokg_seed$s.json   train_central.py --prep $PREP --seed $s --no_kg
  run fl_nodp_seed$s.json        fl_flower.py --seed $s --dp off $FL
  run fl_dp_seed$s.json          fl_flower.py --seed $s --dp on --sigma $SIGMA $FL
  run fl_dp_nokg_seed$s.json     fl_flower.py --seed $s --dp on --sigma $SIGMA $FL --no_kg
  run mia_central_seed$s.json    mia.py --prep $PREP --weights results/central_seed$s.pt --out results/mia_central_seed$s.json --seed $s
  run mia_fl_dp_seed$s.json      mia.py --prep $PREP --weights results/fl_dp_seed$s.pt --dp_model --out results/mia_fl_dp_seed$s.json --seed $s
  run fid_fl_dp_seed$s.json      explain_eval.py --prep $PREP --weights results/fl_dp_seed$s.pt --dp_model --ref_weights results/central_seed$s.pt --out results/fid_fl_dp_seed$s.json --seed $s
  run fid_central_seed$s.json    explain_eval.py --prep $PREP --weights results/central_seed$s.pt --out results/fid_central_seed$s.json --seed $s
done
python stats.py results --ref fl_dp --others central fl_nodp fl_dp_nokg rf --metric f1 | tee results/stats.txt
python make_tables.py results
