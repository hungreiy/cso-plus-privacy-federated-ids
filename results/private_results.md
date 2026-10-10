
### Private non-deep baselines (CIC-IDS2017, seeds 0-4)

| Method | Privacy | n seeds | F1 | F1 clear | AUC | FPR | Recall known | Recall unseen | ε (busiest tenant) |
|---|---|---|---|---|---|---|---|---|---|
| Random forest, pooled data (non-private reference) | none | 5 | 0.8047 ± 0.0008 | 0.9783 ± 0.0011 | 0.9131 ± 0.0010 | 0.0081 ± 0.0004 | 0.9239 ± 0.0010 | 0.4026 ± 0.0021 | - |
| Random forests, local data only (one per tenant) | no data shared | 5 | 0.7866 ± 0.0005 | 0.9750 ± 0.0004 | 0.9015 ± 0.0005 | 0.0074 ± 0.0001 | 0.8917 ± 0.0004 | 0.3834 ± 0.0015 | - |
| Logistic regression, DP-FL (window statistics) | DP | 5 | 0.5638 ± 0.0096 | 0.8678 ± 0.0083 | 0.8237 ± 0.0053 | 0.0012 ± 0.0001 | 0.6916 ± 0.0121 | 0.0453 ± 0.0088 | 6.65 ± 0.38 |
| Proposed: DP-FL hybrid model with KG | DP | 5 | 0.5718 ± 0.0040 | 0.9015 ± 0.0044 | 0.7975 ± 0.0078 | 0.0007 ± 0.0001 | 0.7420 ± 0.0070 | 0.0024 ± 0.0013 | 6.64 ± 0.41 |

### Local-only random forests: spread over tenants (F1 on all windows, mean over seeds)

| Statistic | F1 | F1 clear |
|---|---|---|
| Worst tenant | 0.7809 | 0.9717 |
| Mean of tenants | 0.7866 | 0.9750 |
| Best tenant | 0.7905 | 0.9776 |

Learning rates chosen on validation: seed 0: 1.0, seed 1: 1.0, seed 2: 1.0, seed 3: 1.0, seed 4: 1.0
