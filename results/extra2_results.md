
### Architecture baselines under the same DP-FL (seeds 0-2)

| Architecture | n seeds | F1 | F1 clear | Recall known | AUC | ε (busiest tenant) | Model size (MB) |
|---|---|---|---|---|---|---|---|
| Hybrid CNN-LSTM (proposed) | 3 | 0.5715 ± 0.0053 | 0.9007 ± 0.0058 | 0.7408 ± 0.0090 | 0.7972 ± 0.0077 | 6.66 ± 0.58 | 3.51 ± 0.00 |
| MLP autoencoder-classifier | 3 | 0.5678 ± 0.0075 | 0.8958 ± 0.0088 | 0.7290 ± 0.0118 | 0.8346 ± 0.0089 | 6.42 ± 0.19 | 8.12 ± 0.00 |
| Transformer autoencoder-classifier | 3 | 0.5674 ± 0.0124 | 0.8910 ± 0.0163 | 0.7301 ± 0.0231 | 0.8301 ± 0.0105 | 6.54 ± 0.38 | 0.52 ± 0.00 |

### Membership inference: offline LiRA with shadow models vs global threshold attack

| Target | Attack | n target seeds | AUC | TPR at 1 % FPR | TPR at 0.1 % FPR |
|---|---|---|---|---|---|
| Centralised (non-private) | LiRA (offline, shadow models) | 3 | 0.5341 ± 0.0123 | 0.0130 ± 0.0018 | 0.0013 ± 0.0010 |
| Centralised (non-private) | Global threshold | 3 | 0.5179 ± 0.0053 | 0.0190 ± 0.0048 | 0.0018 ± 0.0013 |
| Proposed DP-FL | LiRA (offline, shadow models) | 3 | 0.5092 ± 0.0102 | 0.0137 ± 0.0078 | 0.0017 ± 0.0025 |
| Proposed DP-FL | Global threshold | 3 | 0.5287 ± 0.0018 | 0.0133 ± 0.0026 | 0.0012 ± 0.0008 |

Shadows per run: 8 (trained 10 epochs each); candidates: 2000 members and 2000 non-members.
