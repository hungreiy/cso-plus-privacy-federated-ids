# CSO+ : Ontology-Driven, Privacy-Aware Federated Intrusion Detection

Code, ontology, and result files for the PhD thesis

> **An Ontology-Driven Deep Learning Approach for Explainable and Privacy-Aware Intrusion Detection for Multi-Tenant Cloud Environments**
> Nor Haizi Azreen Binti Nor Haizily, Faculty of Computer and Mathematical Sciences, Universiti Teknologi MARA (UiTM), 2026–2027.
> Supervisors: Assoc. Prof. Dr. Mohamad Yusof Darus and Assoc. Prof. Dr. Norhaslinda Kamaruddin.

The detector is a hybrid CNN-LSTM autoencoder-classifier over windows of 15 network flows, enriched with 32-dimensional Node2Vec embeddings of a cloud-security ontology (CSO+), trained with federated averaging (Flower) and record-level differential privacy (DP-SGD, Opacus), and explained with SHAP.

## Repository layout

| Folder | Contents |
|---|---|
| `ontology/` | `cso_plus.ttl` (the CSO+ ontology), the script that builds it, a structural check, the Node2Vec embeddings, the shuffled-ontology control, and the SPARQL competency questions |
| `src/` | Data preparation, models, federated training, baselines, explanations, privacy accounting, membership-inference attacks, and table scripts |
| `scripts/` | One-command runners for every group of experiments (see below) |
| `results/` | Result files (JSON) and summary tables (Markdown) for every experiment; model weights are not included |

## Data

The datasets are **not** redistributed. The runners download CIC-IDS2017 (MachineLearningCVE release) and check its SHA-256 checksum; CSE-CIC-IDS2018 is obtained from the Canadian Institute for Cybersecurity. Please cite the dataset authors when using them.

## Hardware and cost

All experiments ran on RunPod with one NVIDIA RTX 2000 Ada (16 GB) or RTX 4000 Ada (20 GB) GPU. One private federated run (50 rounds, 10 tenants) takes about 30–40 minutes; the full set of experiments took several days of GPU time at about USD 0.24 per hour.

## Experiments and thesis sections

| Runner | Experiments | Thesis section |
|---|---|---|
| `run_all.sh` | Main study on CIC-IDS2017 (ten seeds), baselines, ablations, explanations, threshold membership inference | 4.3–4.9 |
| `run_extra.sh` | Non-IID tenants, CSE-CIC-IDS2018, cross-dataset test, window length | 4.10.1–4.10.4 |
| `run_more.sh` | Random-embedding control, privacy-utility curve, additional data splits | 4.10.5–4.10.7 |
| `run_private.sh` | Private non-deep baselines (local random forests, federated DP logistic regression) | 4.10.8 |
| `run_mal.sh` | Malicious tenants (label flipping) with FedAvg and FedMedian | 4.10.9 |
| `run_label.sh` | Label-scarce tenants | 4.10.10 |
| `run_extra2.sh` | Architecture baselines (MLP, Transformer); offline LiRA membership inference | 4.10.11, 4.6 |
| `run_extra3.sh` | Tenant-level DP-FedAvg, shuffled-ontology control, 50 and 100 tenants, model-replacement attacks | 4.10.12, 4.10.5, 4.10.14, 4.10.9 |
| `ontology/ontology_cq.py` | Ten competency questions posed to CSO+ as SPARQL queries | 4.10.13 |

Each runner sets up the environment, runs a smoke test on synthetic data, runs the experiments (skipping finished ones, so it can be restarted), and writes a results archive.

```bash
cd /workspace && nohup bash thesis_experiments/scripts/run_more.sh > /workspace/more.log 2>&1 &
```

## Use of generative AI

A generative AI assistant (Claude, Anthropic) was used to help draft and edit text, check references, write and debug the experiment code, and prepare figures, as declared in the thesis. All research decisions and results are the author's responsibility.

## Citation

See `CITATION.cff`. Please also cite the two ISIEA 2026 papers listed in the thesis.

## Licence

Code: MIT licence (`LICENSE`). Ontology (`ontology/cso_plus.ttl`): CC BY 4.0.
