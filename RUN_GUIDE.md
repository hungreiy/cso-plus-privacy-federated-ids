# Running the experiments (RunPod pod, CIC-IDS2017, hybrid supervised detector = "option B")

## What the detector is
Hybrid CNN-LSTM autoencoder-classifier on 15-flow windows with knowledge-graph (CSO+) embeddings fused at three levels.
Loss = reconstruction error (benign windows only) + attack/benign cross-entropy on the latent code. Alert = attack score above the
99.5th percentile of the benign validation scores (about 0.5 % false alarms). Federated training (Flower) with record-level DP-SGD (Opacus).

## Data split (Section 3.5 / 3.9)
* windows are cut into contiguous blocks of 200; whole blocks go to train (60 %) / validation (10 %) / test (30 %)
* attack types DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Bot, Web Attack (3 types), Infiltration, Heartbleed are UNSEEN:
  they never appear in train or validation, only in the test set. Change with `--holdout` in data.py.
* tenants: the training windows are split at random across 10 simulated tenants

## Files
| Path | Purpose |
|---|---|
| `ontology/` | CSO+ (Turtle), Node2Vec embeddings (`cso_plus_node2vec.npz`), build script, counts |
| `src/data.py` | CSVs -> windows, block split, unseen types, KG vectors, tenants |
| `src/model.py`, `train_central.py`, `fl_flower.py` | model, centralised baseline, Flower + Opacus |
| `src/train_rf.py` | random-forest reference (same split, same threshold rule) |
| `src/choose_sigma.py`, `final_epsilon.py`, `accountant.py`, `preview_epsilon.py` | privacy budget |
| `src/mia.py`, `explain_eval.py`, `shap_explain.py` | membership inference, explanation fidelity, example explanations |
| `src/stats.py`, `make_tables.py` | Wilcoxon + Holm, tables for Chapter 4 (`results/tables.md`) |
| `run_smoke.sh`, `run_pilot.sh`, `run_all.sh` | smoke test (fake data) / pilot (real data) / full study |

## Status (state this honestly in the thesis)
* Tested in my environment: data preparation and metrics on the real data (4 of the 8 files), random forest, table maker.
* NOT tested until you run the smoke test: model, centralised and federated training, DP, MIA, SHAP (PyTorch is not available where I work).
* `rec_weight = 10` (weight of the reconstruction term) and the learning rate 0.001 are not tuned.

## Steps on the pod (Python 3.11 environment, NVIDIA GPU that is NOT Blackwell: no "PRO" cards, no RTX 5090)
1. Environment (after every pod restart):
```
export UV_CACHE_DIR=/tmp/uv-cache
pip install uv && uv venv --python 3.11 /root/venv311 && source /root/venv311/bin/activate
cd /workspace/thesis_experiments && uv pip install --link-mode=copy -r requirements.txt
```
2. Data on the pod's own disk (big files on /workspace can fail):
```
mkdir -p /root/data && cd /root/data && curl -L -o MachineLearningCSV.zip "https://huggingface.co/datasets/Mireu-Lab/CIC-IDS/resolve/main/ZIP/MachineLearningCSV.zip"
sha256sum MachineLearningCSV.zip | cut -c1-32     # must print c3f26274b36c837ccf28ffd2dbf45829
unzip -o -q MachineLearningCSV.zip -d /root/data && ls /root/data/MachineLearningCVE | wc -l    # 8
```
3. `cd /workspace/thesis_experiments && bash run_smoke.sh` (5-8 min). Every line must say OK.
4. `bash run_pilot.sh 2>&1 | tee /workspace/pilot.log` (25-35 min). Send `tail -40 /workspace/pilot.log`.
5. `nohup bash run_all.sh > /workspace/run_all.log 2>&1 &` (5 seeds, about 10 h). Check: `tail -3 /workspace/run_all.log`.
   Copy `results/` to your laptop before stopping the pod. `/root` is erased when the container is rebuilt.

## Known limits to state in Chapter 4
* The ontology relations were written by the author knowing the attack types in CIC-IDS2017 (the no-KG ablation measures any advantage).
* Tenants are a random partition of one dataset; CIC-IDS2017 has no cloud-native attacks.
* Unseen attack types are only detectable through what the classifier generalises (and the reconstruction branch): expect low recall on them.
* A window is an attack if any of its 15 flows is one; the original CIC-IDS2017 release has known labelling errors (Engelen et al., 2021).
* Rare classes (Infiltration 36 flows, Heartbleed 11, SQL injection 21) are too small for reliable per-class numbers.

## Tuning the private model with the validation set only, and lower noise (added after the five-seed study)
Purpose: check whether the private federated model improves when the reconstruction weight and the learning rate are chosen on the
VALIDATION set (never the test set), and when the noise is lowered so that epsilon of the busiest tenant is close to 8.
```
source /root/venv311/bin/activate && cd /workspace/thesis_experiments/src
# data must be prepared first (same as run_all.sh): /root/prepared with meta.json
python tune_dp.py --stage 1 --dry_run                       # prints sigma and the commands, runs nothing
nohup python tune_dp.py --stage all > /workspace/tune.log 2>&1 &   # about 7-8 h, resumable
tail -3 /workspace/tune.log
```
* Stage 1: 4 reconstruction weights (0, 1, 10, 30) x 3 learning rates (3e-4, 1e-3, 3e-3), 20 rounds, seed 100, `--skip_test` (test set never evaluated).
* Stage 2: the best 3 at 50 rounds. Selection metric: validation average precision. The choice is written to `results_tuned/best.json`.
* Stage 3: seeds 0-4 with the chosen values: DP-FL with KG, DP-FL without KG, FedAvg without DP, membership inference, explanation fidelity, tables in `results_tuned/final/tables.md`.
* Noise: a tenant trains at most `--cap 20` rounds (a rule that does not look at data); sigma is computed so that epsilon at 20 rounds is 8.0 (about 1.29 instead of 1.455).
  The busiest tenant therefore has epsilon of at most 8.0 and about 7.6 to 8.0 in practice.
* Expectation (honest): lowering sigma from 1.455 to about 1.29 is a small change; the gain may be small or zero. Report whatever you get.
