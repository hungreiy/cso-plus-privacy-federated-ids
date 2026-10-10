#!/usr/bin/env bash
# Pilot on the REAL data (about 25-35 min). Prepares the data, then: random forest reference, 10 epochs of the centralised
# hybrid model, and 3 rounds of DP federated training. Prints metrics and the time of each step. Writes to results_pilot/.
set -e
cd "$(dirname "$0")/src"
CSV=/root/data/MachineLearningCVE; KG=../ontology/cso_plus_node2vec.npz
[ -f /root/prepared/meta.json ] || python data.py --csv_dir $CSV --kg $KG --out_dir /root/prepared --n_tenants 10 --partition iid
SIGMA=$(python choose_sigma.py --prep /root/prepared --rounds 50 --fraction 0.3 --quiet)
echo "noise multiplier for target epsilon 8 (worst tenant): $SIGMA"
S=$(date +%s); python train_rf.py --prep /root/prepared --out results_pilot --trees 100 --seed 0 2>&1 | tail -1; echo "random forest: $(( $(date +%s) - S )) s"
S=$(date +%s); python train_central.py --prep /root/prepared --out results_pilot --epochs 10 --seed 0 2>&1 | grep --line-buffered -E "^epoch|^\{"; echo "centralised, 10 epochs: $(( $(date +%s) - S )) s"
S=$(date +%s); python fl_flower.py --prep /root/prepared --out results_pilot --dp on --sigma $SIGMA --clients 10 --rounds 3 --fraction 0.3 --local_epochs 5 --eval_every 1 --seed 0 2>&1 | grep -E "^round|^\{"; echo "DP federated, 3 rounds: $(( $(date +%s) - S )) s (full run = this x 50/3)"
