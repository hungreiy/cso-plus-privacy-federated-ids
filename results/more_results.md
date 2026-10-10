
### Random-embedding control (DP-FL, CIC-IDS2017)

| Model | n seeds | F1 | F1 clear | AUC | Recall known | Recall unseen |
|---|---|---|---|---|---|---|
| DP-FL with CSO+ embeddings (main) | 10 | 0.5738 ± 0.0053 | 0.9038 ± 0.0053 | 0.7987 ± 0.0085 | 0.7449 ± 0.0082 | 0.0033 ± 0.0025 |
| DP-FL with random embeddings (control) | 10 | 0.5565 ± 0.0200 | 0.8842 ± 0.0250 | 0.8261 ± 0.0141 | 0.7150 ± 0.0350 | 0.0026 ± 0.0014 |
| DP-FL without embeddings | 10 | 0.5466 ± 0.0074 | 0.8723 ± 0.0091 | 0.7947 ± 0.0104 | 0.6978 ± 0.0127 | 0.0017 ± 0.0007 |

### Paired comparisons (Wilcoxon signed-rank, two-sided)

| A | B | Metric | n | Mean A-B | A > B | p |
|---|---|---|---|---|---|---|
| DP-FL with CSO+ embeddings (main) | DP-FL with random embeddings (control) | F1 | 10 | +0.0174 | 9/10 | 0.0059 |
| DP-FL with CSO+ embeddings (main) | DP-FL with random embeddings (control) | F1 clear | 10 | +0.0196 | 9/10 | 0.0059 |
| DP-FL with random embeddings (control) | DP-FL without embeddings | F1 | 10 | +0.0098 | 8/10 | 0.1055 |
| DP-FL with random embeddings (control) | DP-FL without embeddings | F1 clear | 10 | +0.0119 | 8/10 | 0.1055 |
| DP-FL with CSO+ embeddings (main) | DP-FL without embeddings | F1 | 10 | +0.0272 | 10/10 | 0.0020 |

### Privacy-utility curve (DP-FL with CSO+ embeddings, seeds 0-4)

| Target ε | σ | Achieved ε (busiest tenant) | n seeds | F1 | F1 clear | AUC | Recall known | Recall unseen |
|---|---|---|---|---|---|---|---|---|
| no DP (FedAvg) | - | - | 5 | 0.6784 ± 0.0072 | 0.9104 ± 0.0032 | 0.8168 ± 0.0080 | 0.8887 ± 0.0036 | 0.0843 ± 0.0146 |
| 1 | 8.296 | 0.86 ± 0.07 | 5 | 0.5545 ± 0.0083 | 0.8821 ± 0.0100 | 0.7932 ± 0.0230 | 0.7126 ± 0.0145 | 0.0007 ± 0.0008 |
| 2 | 4.413 | 1.70 ± 0.08 | 5 | 0.5644 ± 0.0065 | 0.8943 ± 0.0080 | 0.7968 ± 0.0251 | 0.7290 ± 0.0120 | 0.0020 ± 0.0010 |
| 4 | 2.435 | 3.35 ± 0.24 | 5 | 0.5670 ± 0.0071 | 0.8965 ± 0.0091 | 0.7911 ± 0.0076 | 0.7337 ± 0.0128 | 0.0020 ± 0.0014 |
| 8 (main) | 1.455 | 6.64 ± 0.41 | 5 | 0.5718 ± 0.0040 | 0.9015 ± 0.0044 | 0.7975 ± 0.0078 | 0.7420 ± 0.0070 | 0.0024 ± 0.0013 |

### F1-score on all test windows by data split (seeds 0-2 per split; last column = mean ± SD across splits)

| Method | Split 0 (main) | Split 1 | Split 2 | Across splits |
|---|---|---|---|---|
| Random forest | 0.8047 ± 0.0009 | 0.7968 ± 0.0021 | 0.8048 ± 0.0024 | 0.8021 ± 0.0046 |
| Centralised hybrid | 0.6833 ± 0.0052 | 0.6570 ± 0.0130 | 0.6946 ± 0.0099 | 0.6783 ± 0.0193 |
| FedAvg without DP | 0.6751 ± 0.0051 | 0.6508 ± 0.0057 | 0.6829 ± 0.0035 | 0.6696 ± 0.0167 |
| Proposed DP-FL with KG | 0.5715 ± 0.0053 | 0.5640 ± 0.0032 | 0.6153 ± 0.0024 | 0.5836 ± 0.0277 |
| DP-FL without KG | 0.5417 ± 0.0051 | 0.5504 ± 0.0082 | 0.5921 ± 0.0080 | 0.5614 ± 0.0270 |

### Ranking of the methods by F1-score in each split

| Split | Ranking |
|---|---|
| Split 0 (main) | Random forest > Centralised hybrid > FedAvg without DP > Proposed DP-FL with KG > DP-FL without KG |
| Split 1 | Random forest > Centralised hybrid > FedAvg without DP > Proposed DP-FL with KG > DP-FL without KG |
| Split 2 | Random forest > Centralised hybrid > FedAvg without DP > Proposed DP-FL with KG > DP-FL without KG |
