#!/usr/bin/env bash
# End-to-end check on FAKE data (about 5-8 minutes). One OK/FAIL line per step. Outputs are NOT results.
cd "$(dirname "$0")/src"
FAILS=0
step() { name=$1; shift; if "$@" > /tmp/smoke_$name.log 2>&1; then echo "OK    $name"; else echo "FAIL  $name   -> tail -30 /tmp/smoke_$name.log"; FAILS=$((FAILS+1)); fi; }
rm -rf smoke_csv prepared_smoke results_smoke
step make_data      python make_smoke_data.py smoke_csv
step prepare_data   python data.py --csv_dir smoke_csv --kg smoke_csv/SMOKE_ONLY_kg_embeddings.npz --out_dir prepared_smoke --n_tenants 5 --block 10
step central        python train_central.py --prep prepared_smoke --out results_smoke --epochs 2
step central_nokg   python train_central.py --prep prepared_smoke --out results_smoke --epochs 2 --no_kg
step fl_nodp        python fl_flower.py --prep prepared_smoke --out results_smoke --dp off --clients 5 --rounds 2 --local_epochs 1 --eval_every 1 --fraction 1.0
step fl_dp          python fl_flower.py --prep prepared_smoke --out results_smoke --dp on --sigma 1.0 --clients 5 --rounds 2 --local_epochs 1 --eval_every 1 --fraction 1.0
step epsilon        python final_epsilon.py results_smoke/dp_log_fl_dp_seed0.jsonl
step rf             python train_rf.py --prep prepared_smoke --out results_smoke --trees 20
step mia_central    python mia.py --prep prepared_smoke --weights results_smoke/central_seed0.pt --out results_smoke/mia_central_seed0.json --n 100
step mia_dp         python mia.py --prep prepared_smoke --weights results_smoke/fl_dp_seed0.pt --dp_model --out results_smoke/mia_fl_dp_seed0.json --n 100
step explain_central python explain_eval.py --prep prepared_smoke --weights results_smoke/central_seed0.pt --n_alerts 20 --out results_smoke/fid_central_seed0.json
step explain_dp     python explain_eval.py --prep prepared_smoke --weights results_smoke/fl_dp_seed0.pt --dp_model --ref_weights results_smoke/central_seed0.pt --n_alerts 20 --out results_smoke/fid_fl_dp_seed0.json
step tables         python make_tables.py results_smoke
echo "--- epsilon (my accountant vs Opacus):"; grep -E "eps_" /tmp/smoke_epsilon.log
if [ $FAILS -eq 0 ]; then echo "SMOKE TEST PASSED"; else echo "SMOKE TEST: $FAILS step(s) FAILED"; fi
