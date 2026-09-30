# Ontology-driven, privacy-aware hybrid intrusion detector (thesis code)

Code, ontology, and result files for the thesis "An Ontology-Driven Deep Learning Approach for Explainable and Privacy-Aware Intrusion Detection in Cloud Computing".

* `ontology/` CSO+ ontology (Turtle), Node2Vec port embeddings, build script, and `check_cso_plus.py` (structural checks and competency questions; not a reasoner).
* `src/` data preparation (block split, held-out attack types), model, centralised and federated training (Flower + Opacus), random-forest reference, membership inference, SHAP explanations, tables.
* `results/` per-seed result files (JSON, model weights excluded), `tables.md`, `stats.txt`, `shap_examples.json` of the five-seed run reported in Chapter 4.
* `run_smoke.sh`, `run_pilot.sh`, `run_all.sh`, `RUN_GUIDE.md` how to reproduce (Python 3.11, PyTorch 2.5.1, Opacus 1.5.2, Flower 1.11.1; a GPU that is not Blackwell).

Data: CIC-IDS2017 (MachineLearningCVE release) must be downloaded from its source; it is not included. Seeds 0 to 4, split seed 0.
Status: research code, not production code. See Chapter 4.11 of the thesis for limitations.
