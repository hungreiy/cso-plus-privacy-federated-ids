#!/usr/bin/env bash
# Additional experiments for the thesis. Resumable: finished runs are skipped. Run `bash run_smoke_extra.sh` first.
#   nohup bash run_extra.sh all > /workspace/extra.log 2>&1 &          (or choose stages, e.g.  bash run_extra.sh seeds baselines)
# Stages (in this order for "all"), rough time on an RTX 4000 Ada:
#   seeds      A2  CIC-IDS2017 main study, seeds 5-9 (+ Wilcoxon test over 10 seeds)          ~10 h
#   baselines  A3  XGBoost, classifier-only, autoencoder-only on CIC-IDS2017, seeds 0-9        ~2 h
#   noniid     A4  non-IID tenants (Dirichlet 0.5): FedAvg, DP-FL, DP-FL without KG, seeds 0-4 ~4.5 h
#   ds2018     A1  download CSE-CIC-IDS2018, same study on it (7 methods), seeds 0-4           ~9 h (+ download)
#   cross      A1  trained on CIC-IDS2017, tested unchanged on CSE-CIC-IDS2018, seeds 5-9      ~1 h (uses the models of the seeds stage)
#   window     A5  window length 5 and 30: random forest, centralised, DP-FL, seeds 0-2        ~5 h
set -e
cd "$(dirname "$0")/src"
CSV=/root/data/MachineLearningCVE; CSV18=/root/data18; KG=../ontology/cso_plus_node2vec.npz
TENANTS=10; ROUNDS=50; EPOCHS=5; FRACTION=0.3; TARGET_EPS=8.0
STAGES="$*"; [ -z "$STAGES" ] || [ "$STAGES" = "all" ] && STAGES="seeds baselines noniid ds2018 cross window"
run() { out=$1; shift; [ -f "$out" ] || python "$@"; }
sigma() { python choose_sigma.py --prep $1 --rounds $ROUNDS --fraction $FRACTION --local_epochs $EPOCHS --target_eps $TARGET_EPS --quiet; }
fl() { echo "--prep $1 --out $2 --clients $TENANTS --rounds $ROUNDS --local_epochs $EPOCHS --fraction $FRACTION"; }
prep17() {   # (re)build the main CIC-IDS2017 prep if missing or built by the old data.py (no scaler.npz); same seed -> same split
  [ -f /root/prepared/scaler.npz ] || { rm -rf /root/prepared; python data.py --csv_dir $CSV --kg $KG --out_dir /root/prepared --n_tenants $TENANTS --partition iid; }
}
for st in $STAGES; do echo "################ stage $st ($(date))"; case $st in
seeds)
  prep17; SEEDS="5 6 7 8 9" bash ../run_all.sh ;;
baselines)
  prep17
  for s in 0 1 2 3 4 5 6 7 8 9; do
    run results/xgb_seed$s.json      train_xgb.py --prep /root/prepared --out results --seed $s
    run results/cls_only_seed$s.json train_central.py --prep /root/prepared --out results --seed $s --rec_weight 0 --tag cls_only
    run results/ae_only_seed$s.json  train_central.py --prep /root/prepared --out results --seed $s --mode ae_only --tag ae_only
  done
  python stats.py results --ref fl_dp --others central fl_nodp fl_dp_nokg rf xgb cls_only ae_only --metric f1 | tee results/stats.txt
  python make_tables.py results ;;
noniid)
  P=/root/prepared_noniid; R=results_noniid; mkdir -p $R
  [ -f $P/meta.json ] || python data.py --csv_dir $CSV --kg $KG --out_dir $P --n_tenants $TENANTS --partition dirichlet --alpha 0.5
  SIG=$(sigma $P); echo "non-IID sigma (smallest tenant): $SIG"
  for s in 0 1 2 3 4; do
    run $R/fl_nodp_seed$s.json    fl_flower.py --seed $s --dp off $(fl $P $R)
    run $R/fl_dp_seed$s.json      fl_flower.py --seed $s --dp on --sigma $SIG $(fl $P $R)
    run $R/fl_dp_nokg_seed$s.json fl_flower.py --seed $s --dp on --sigma $SIG $(fl $P $R) --no_kg
  done
  python make_tables.py $R ;;
ds2018)
  if [ ! -f $CSV18/.done ]; then
    mkdir -p $CSV18; pip install -q awscli || uv pip install -q awscli
    aws s3 sync --no-sign-request "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" $CSV18/ && touch $CSV18/.done
  fi
  python -c "import ids2018,glob; fs=sorted(glob.glob('$CSV18/*.csv')); assert fs, 'no 2018 CSV files'
for f in fs:
    u,m=ids2018.check_columns(f); print(f.split('/')[-1], 'OK' if not u and not m else f'unmapped={u} missing={m}'); assert not u and not m"
  P=/root/prepared18; R=results_2018; mkdir -p $R
  [ -f $P/meta.json ] || python data.py --csv_dir $CSV18 --kg $KG --out_dir $P --n_tenants $TENANTS --partition iid --dataset cic2018 --max_windows 250000 --cap_per_type 20000
  SIG=$(sigma $P); echo "2018 sigma: $SIG"
  for s in 0 1 2 3 4; do
    run $R/rf_seed$s.json           train_rf.py --prep $P --out $R --seed $s
    run $R/xgb_seed$s.json          train_xgb.py --prep $P --out $R --seed $s
    run $R/central_seed$s.json      train_central.py --prep $P --out $R --seed $s
    run $R/central_nokg_seed$s.json train_central.py --prep $P --out $R --seed $s --no_kg
    run $R/fl_nodp_seed$s.json      fl_flower.py --seed $s --dp off $(fl $P $R)
    run $R/fl_dp_seed$s.json        fl_flower.py --seed $s --dp on --sigma $SIG $(fl $P $R)
    run $R/fl_dp_nokg_seed$s.json   fl_flower.py --seed $s --dp on --sigma $SIG $(fl $P $R) --no_kg
  done
  python stats.py $R --ref fl_dp --others central fl_nodp fl_dp_nokg rf xgb --metric f1 | tee $R/stats.txt
  python make_tables.py $R ;;
cross)
  prep17; [ -f $CSV18/.done ] || { echo "run the ds2018 stage first (it downloads CSE-CIC-IDS2018)"; exit 1; }
  for s in 0 1 2 3 4; do run results_cross/cross_fl_dp_seed$s.json cross_eval.py --prep17 /root/prepared --results17 results --csv18 $CSV18 --kg $KG --seed $s --out results_cross; done ;;
window)
  for W in 5 30; do
    P=/root/prepared_w$W; R=results_w$W; mkdir -p $R; B=$((3000 / W))          # same block length in flows (3000) as the main study
    [ -f $P/meta.json ] || python data.py --csv_dir $CSV --kg $KG --out_dir $P --n_tenants $TENANTS --partition iid --window $W --block $B
    SIG=$(sigma $P); echo "window $W sigma: $SIG"
    for s in 0 1 2; do
      run $R/rf_seed$s.json      train_rf.py --prep $P --out $R --seed $s
      run $R/central_seed$s.json train_central.py --prep $P --out $R --seed $s
      run $R/fl_dp_seed$s.json   fl_flower.py --seed $s --dp on --sigma $SIG $(fl $P $R)
    done
  done ;;
*) echo "unknown stage $st"; exit 1 ;;
esac; done
python make_extra_tables.py . || true
echo "################ all requested stages finished ($(date))"
