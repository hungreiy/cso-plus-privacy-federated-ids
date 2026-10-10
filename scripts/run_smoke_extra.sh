#!/usr/bin/env bash
# Check of the ADDITIONAL experiments on FAKE data (about 5-8 minutes). One OK/FAIL line per step. Outputs are NOT results.
cd "$(dirname "$0")/src"
FAILS=0
step() { name=$1; shift; if "$@" > /tmp/smokex_$name.log 2>&1; then echo "OK    $name"; else echo "FAIL  $name   -> tail -30 /tmp/smokex_$name.log"; FAILS=$((FAILS+1)); fi; }
S=smoke_extra; rm -rf $S; mkdir -p $S
KG=$S/SMOKE_ONLY_kg.npz; C="--block 10 --holdout Bot"
step make_data       python make_smoke_extra.py $S
step prep_iid        python data.py --csv_dir $S/csv17 --kg $KG --out_dir $S/p --n_tenants 4 $C
step prep_noniid     python data.py --csv_dir $S/csv17 --kg $KG --out_dir $S/pn --n_tenants 4 --partition dirichlet --alpha 0.5 --min_tenant 20 $C
step prep_w5         python data.py --csv_dir $S/csv17 --kg $KG --out_dir $S/p5 --n_tenants 4 --window 5 --block 30 --holdout Bot
step prep_w30        python data.py --csv_dir $S/csv17 --kg $KG --out_dir $S/p30 --n_tenants 4 --window 30 --block 5 --holdout Bot
step check_2018_cols python -c "import ids2018,glob; r=[ids2018.check_columns(f) for f in glob.glob('$S/csv18/*.csv')]; assert all(u==[] and m==[] for u,m in r), r"
step prep_2018       python data.py --csv_dir $S/csv18 --kg $KG --out_dir $S/p18 --n_tenants 4 --dataset cic2018 --block 10 --max_windows 900 --cap_per_type 150
step central         python train_central.py --prep $S/p --out $S/r --epochs 2
step cls_only        python train_central.py --prep $S/p --out $S/r --epochs 2 --rec_weight 0 --tag cls_only
step ae_only         python train_central.py --prep $S/p --out $S/r --epochs 2 --mode ae_only --tag ae_only
step rf              python train_rf.py --prep $S/p --out $S/r --trees 20
step xgb             python train_xgb.py --prep $S/p --out $S/r --trees 30
step fl_dp           python fl_flower.py --prep $S/p --out $S/r --dp on --sigma 1.0 --clients 4 --rounds 2 --local_epochs 1 --eval_every 1 --fraction 1.0
step fl_dp_noniid    python fl_flower.py --prep $S/pn --out $S/rn --dp on --sigma 1.0 --clients 4 --rounds 2 --local_epochs 1 --eval_every 1 --fraction 1.0
step central_w5      python train_central.py --prep $S/p5 --out $S/r5 --epochs 1
step central_w30     python train_central.py --prep $S/p30 --out $S/r30 --epochs 1
step fl_dp_w5        python fl_flower.py --prep $S/p5 --out $S/r5 --dp on --sigma 1.0 --clients 4 --rounds 1 --local_epochs 1 --eval_every 1 --fraction 1.0
step central_2018    python train_central.py --prep $S/p18 --out $S/r18 --epochs 1
step sigma_noniid    python choose_sigma.py --prep $S/pn --rounds 50 --fraction 0.3 --local_epochs 5 --target_eps 8 --quiet
step cross           python cross_eval.py --prep17 $S/p --results17 $S/r --csv18 $S/csv18 --kg $KG --out $S/rc
step tables_main     python make_tables.py $S/r
step tables_extra    bash -c "mkdir -p $S/x && ln -sfn ../r $S/x/results && ln -sfn ../rn $S/x/results_noniid && ln -sfn ../r5 $S/x/results_w5 && ln -sfn ../r30 $S/x/results_w30 && ln -sfn ../rc $S/x/results_cross && python make_extra_tables.py $S/x"
grep -h "^{" /tmp/smokex_cross.log | head -1
if [ $FAILS -eq 0 ]; then echo "EXTRA SMOKE TEST PASSED"; else echo "EXTRA SMOKE TEST: $FAILS step(s) FAILED"; fi
