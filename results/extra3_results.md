
### Tenant-level DP-FedAvg (sigma 1.74, 10 tenants, 3 per round, 50 rounds, seeds 0-2)

| Update clip norm | n | F1 | F1 clear | Recall known | AUC | ε |
|---|---|---|---|---|---|---|
| 0.5 | 3 | 0.0072 ± 0.0029 | 0.0055 ± 0.0048 | 0.0020 ± 0.0014 | 0.4771 ± 0.0827 | 8.00 |
| 5 | 3 | 0.0080 ± 0.0030 | 0.0036 ± 0.0016 | 0.0025 ± 0.0003 | 0.4956 ± 0.0601 | 8.00 |
| record-level DP-SGD (main) | 3 | 0.5715 ± 0.0053 | 0.9007 ± 0.0058 | 0.7408 ± 0.0090 | 0.7972 ± 0.0077 | record-level 6.66 ± 0.58 |

### Shuffled-ontology control (DP-FL, seeds 0-9)

| Model | n | F1 | F1 clear | AUC | Recall known |
|---|---|---|---|---|---|
| CSO+ embeddings (main) | 10 | 0.5738 ± 0.0053 | 0.9038 ± 0.0053 | 0.7987 ± 0.0085 | 0.7449 ± 0.0082 |
| Shuffled CSO+ (control) | 10 | 0.5738 ± 0.0118 | 0.9029 ± 0.0132 | 0.8207 ± 0.0214 | 0.7446 ± 0.0202 |
| Random vectors (control) | 10 | 0.5565 ± 0.0200 | 0.8842 ± 0.0250 | 0.8261 ± 0.0141 | 0.7150 ± 0.0350 |
| No embeddings | 10 | 0.5466 ± 0.0074 | 0.8723 ± 0.0091 | 0.7947 ± 0.0104 | 0.6978 ± 0.0127 |

### Paired comparisons on F1 (Wilcoxon signed-rank)

| A | B | n | Mean A-B | A > B | p |
|---|---|---|---|---|---|
| CSO+ embeddings (main) | Shuffled CSO+ (control) | 10 | -0.0000 | 2/10 | 0.3750 |
| Shuffled CSO+ (control) | No embeddings | 10 | +0.0272 | 10/10 | 0.0020 |
| Shuffled CSO+ (control) | Random vectors (control) | 10 | +0.0174 | 9/10 | 0.0098 |

### Scaling the number of tenants (IID, seeds 0-2)

| Tenants | Local RF F1 (mean) | Local RF F1 (weakest tenant) | FedAvg F1 | DP-FL F1 | DP-FL F1 clear | ε (busiest tenant) |
|---|---|---|---|---|---|---|
| 10 (main) | 0.7865 ± 0.0007 | 0.7818 ± 0.0003 | 0.6751 ± 0.0051 | 0.5715 ± 0.0053 | 0.9007 ± 0.0058 | 6.66 ± 0.58 |
| 50 | 0.7530 ± 0.0008 | 0.7089 ± 0.0113 | 0.6654 ± 0.0013 | 0.5583 ± 0.0082 | 0.8869 ± 0.0100 | 7.34 ± 0.35 |
| 100 | 0.7206 ± 0.0014 | 0.6255 ± 0.0071 | 0.6429 ± 0.0068 | 0.5304 ± 0.0086 | 0.8517 ± 0.0110 | 6.54 ± 0.09 |

### Model-replacement attack (label flipping, no DP, update boosted 10x) vs aggregation (seeds 0-2)

| Malicious tenants | Aggregation | n | F1 | F1 clear | Recall known |
|---|---|---|---|---|---|
| 0 | FedAvg | 3 | 0.5715 ± 0.0053 | 0.9007 ± 0.0058 | 0.7408 ± 0.0090 |
| 0 | Norm bounding | 3 | 0.5673 ± 0.0129 | 0.8941 ± 0.0145 | 0.7333 ± 0.0217 |
| 1 | FedAvg | 3 | 0.1100 ± 0.1449 | 0.1989 ± 0.2584 | 0.1113 ± 0.1551 |
| 1 | FedMedian | 3 | 0.5229 ± 0.0337 | 0.8423 ± 0.0430 | 0.6571 ± 0.0561 |
| 1 | Norm bounding | 3 | 0.3287 ± 0.1600 | 0.5645 ± 0.2306 | 0.3765 ± 0.2288 |
| 3 | FedAvg | 3 | 0.1544 ± 0.2290 | 0.2627 ± 0.3790 | 0.1618 ± 0.2444 |
| 3 | FedMedian | 3 | 0.0841 ± 0.1289 | 0.1524 ± 0.2369 | 0.0848 ± 0.1329 |
| 3 | Norm bounding | 3 | 0.0517 ± 0.0588 | 0.0958 ± 0.1164 | 0.0469 ± 0.0572 |
