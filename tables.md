
### Table 4.3 Detection performance on the test set (alert threshold = 99.5th percentile of benign validation scores)

| Method | n seeds | Precision | Recall | F1 | AUC-ROC | FPR |
|---|---|---|---|---|---|---|
| Random forest (non-deep, non-private, reference) | 5 | 0.9781 ± 0.0011 | 0.6835 ± 0.0014 | 0.8047 ± 0.0008 | 0.9131 ± 0.0010 | 0.0081 ± 0.0004 |
| Centralised hybrid model (non-private) | 5 | 0.9924 ± 0.0015 | 0.5180 ± 0.0066 | 0.6807 ± 0.0056 | 0.8289 ± 0.0084 | 0.0021 ± 0.0004 |
| Centralised without KG (ablation) | 5 | 0.9940 ± 0.0008 | 0.5209 ± 0.0070 | 0.6835 ± 0.0061 | 0.8606 ± 0.0055 | 0.0017 ± 0.0002 |
| FedAvg without DP | 5 | 0.9840 ± 0.0021 | 0.5177 ± 0.0085 | 0.6784 ± 0.0072 | 0.8168 ± 0.0080 | 0.0045 ± 0.0006 |
| Proposed: DP-FL with KG | 5 | 0.9968 ± 0.0004 | 0.4009 ± 0.0040 | 0.5718 ± 0.0040 | 0.7975 ± 0.0078 | 0.0007 ± 0.0001 |
| DP-FL without KG (ablation) | 5 | 0.9957 ± 0.0009 | 0.3750 ± 0.0086 | 0.5448 ± 0.0090 | 0.7895 ± 0.0070 | 0.0009 ± 0.0002 |

### Table 4.4 Known (seen) versus unseen attack types

| Method | Recall seen | Recall unseen | AUC seen | AUC unseen | Two-branch recall unseen | AUC (reconstruction only) unseen |
|---|---|---|---|---|---|---|
| Random forest (non-deep, non-private, reference) | 0.9239 ± 0.0010 | 0.4026 ± 0.0021 | 0.9858 ± 0.0002 | 0.8281 ± 0.0021 | - | - |
| Centralised hybrid model (non-private) | 0.8934 ± 0.0047 | 0.0795 ± 0.0178 | 0.9970 ± 0.0004 | 0.6324 ± 0.0181 | 0.1021 ± 0.0166 | 0.6967 ± 0.0035 |
| Centralised without KG (ablation) | 0.8647 ± 0.0025 | 0.1193 ± 0.0135 | 0.9915 ± 0.0005 | 0.7078 ± 0.0121 | 0.1406 ± 0.0134 | 0.6908 ± 0.0067 |
| FedAvg without DP | 0.8887 ± 0.0036 | 0.0843 ± 0.0146 | 0.9954 ± 0.0004 | 0.6081 ± 0.0171 | 0.0973 ± 0.0140 | 0.6905 ± 0.0049 |
| Proposed: DP-FL with KG | 0.7420 ± 0.0070 | 0.0024 ± 0.0013 | 0.9774 ± 0.0039 | 0.5873 ± 0.0157 | 0.0223 ± 0.0025 | 0.6027 ± 0.0036 |
| DP-FL without KG (ablation) | 0.6946 ± 0.0156 | 0.0015 ± 0.0006 | 0.9427 ± 0.0020 | 0.6104 ± 0.0135 | 0.0206 ± 0.0011 | 0.6023 ± 0.0044 |

### Table 4.3b Clear windows only (benign, or at least 8 of the 15 flows are attack flows)

| Method | Precision | Recall | F1 | AUC-ROC | FPR | Recall seen | Recall unseen |
|---|---|---|---|---|---|---|---|
| Random forest (non-deep, non-private, reference) | 0.9691 ± 0.0015 | 0.9877 ± 0.0016 | 0.9783 ± 0.0011 | 0.9988 ± 0.0001 | 0.0081 ± 0.0004 | 0.9986 ± 0.0002 | 0.8549 ± 0.0217 |
| Centralised hybrid model (non-private) | 0.9906 ± 0.0019 | 0.8625 ± 0.0036 | 0.9221 ± 0.0026 | 0.9696 ± 0.0086 | 0.0021 ± 0.0004 | 0.9231 ± 0.0050 | 0.1260 ± 0.0258 |
| Centralised without KG (ablation) | 0.9925 ± 0.0010 | 0.8659 ± 0.0040 | 0.9249 ± 0.0025 | 0.9601 ± 0.0055 | 0.0017 ± 0.0002 | 0.9288 ± 0.0050 | 0.1018 ± 0.0157 |
| FedAvg without DP | 0.9800 ± 0.0028 | 0.8500 ± 0.0035 | 0.9104 ± 0.0032 | 0.9695 ± 0.0041 | 0.0045 ± 0.0006 | 0.9101 ± 0.0038 | 0.1193 ± 0.0036 |
| Proposed: DP-FL with KG | 0.9968 ± 0.0004 | 0.8228 ± 0.0074 | 0.9015 ± 0.0044 | 0.9645 ± 0.0048 | 0.0007 ± 0.0001 | 0.8880 ± 0.0078 | 0.0303 ± 0.0159 |
| DP-FL without KG (ablation) | 0.9957 ± 0.0009 | 0.7727 ± 0.0176 | 0.8700 ± 0.0111 | 0.9731 ± 0.0048 | 0.0009 ± 0.0002 | 0.8347 ± 0.0186 | 0.0189 ± 0.0073 |

### Recall by the number of attack flows in the window (n = test windows)

| Method | seen, 8-15 | seen, 4-7 | seen, 1-3 | unseen, 8-15 | unseen, 4-7 | unseen, 1-3 |
|---|---|---|---|---|---|---|
| Random forest (non-deep, non-private, reference) | 0.999 ± 0.000 (n=9885) | 0.748 ± 0.012 (n=404) | 0.504 ± 0.005 (n=1591) | 0.855 ± 0.022 (n=813) | 0.246 ± 0.006 (n=322) | 0.367 ± 0.002 (n=9033) |
| Centralised hybrid model (non-private) | 0.923 ± 0.005 (n=9885) | 0.862 ± 0.028 (n=404) | 0.717 ± 0.034 (n=1591) | 0.126 ± 0.026 (n=813) | 0.112 ± 0.018 (n=322) | 0.074 ± 0.020 (n=9033) |
| Centralised without KG (ablation) | 0.929 ± 0.005 (n=9885) | 0.839 ± 0.042 (n=404) | 0.473 ± 0.028 (n=1591) | 0.102 ± 0.016 (n=813) | 0.129 ± 0.016 (n=322) | 0.120 ± 0.015 (n=9033) |
| FedAvg without DP | 0.910 ± 0.004 (n=9885) | 0.883 ± 0.022 (n=404) | 0.758 ± 0.037 (n=1591) | 0.119 ± 0.004 (n=813) | 0.135 ± 0.017 (n=322) | 0.079 ± 0.016 (n=9033) |
| Proposed: DP-FL with KG | 0.888 ± 0.008 (n=9885) | 0.092 ± 0.045 (n=404) | 0.000 ± 0.000 (n=1591) | 0.030 ± 0.016 (n=813) | 0.000 ± 0.000 (n=322) | 0.000 ± 0.000 (n=9033) |
| DP-FL without KG (ablation) | 0.835 ± 0.019 (n=9885) | 0.003 ± 0.002 (n=404) | 0.000 ± 0.000 (n=1591) | 0.019 ± 0.007 (n=813) | 0.000 ± 0.000 (n=322) | 0.000 ± 0.000 (n=9033) |

### Per-attack-type recall: Centralised hybrid model (non-private)

| Attack type | Test windows | Unseen in training? | Recall |
|---|---|---|---|
| Bot | 984 | yes | 0.003 ± 0.001 |
| DDoS | 2547 | no | 0.997 ± 0.000 |
| DoS GoldenEye | 6608 | yes | 0.116 ± 0.027 |
| DoS Hulk | 4114 | no | 0.992 ± 0.000 |
| DoS Slowhttptest | 372 | yes | 0.060 ± 0.023 |
| DoS slowloris | 821 | yes | 0.010 ± 0.014 |
| FTP-Patator | 656 | no | 0.841 ± 0.042 |
| Heartbleed | 5 | yes | 0.280 ± 0.179 |
| Infiltration | 35 | yes | 0.000 ± 0.000 |
| PortScan | 3250 | no | 0.756 ± 0.016 |
| SSH-Patator | 1313 | no | 0.748 ± 0.038 |
| Web Attack - Brute Force | 954 | yes | 0.005 ± 0.003 |
| Web Attack - Sql Injection | 20 | yes | 0.000 ± 0.000 |
| Web Attack - XSS | 369 | yes | 0.006 ± 0.001 |

### Per-attack-type recall: Proposed: DP-FL with KG

| Attack type | Test windows | Unseen in training? | Recall |
|---|---|---|---|
| Bot | 984 | yes | 0.000 ± 0.000 |
| DDoS | 2547 | no | 0.992 ± 0.006 |
| DoS GoldenEye | 6608 | yes | 0.002 ± 0.002 |
| DoS Hulk | 4114 | no | 0.955 ± 0.003 |
| DoS Slowhttptest | 372 | yes | 0.028 ± 0.043 |
| DoS slowloris | 821 | yes | 0.000 ± 0.000 |
| FTP-Patator | 656 | no | 0.147 ± 0.041 |
| Heartbleed | 5 | yes | 0.000 ± 0.000 |
| Infiltration | 35 | yes | 0.000 ± 0.000 |
| PortScan | 3250 | no | 0.696 ± 0.018 |
| SSH-Patator | 1313 | no | 0.000 ± 0.001 |
| Web Attack - Brute Force | 954 | yes | 0.000 ± 0.000 |
| Web Attack - Sql Injection | 20 | yes | 0.000 ± 0.000 |
| Web Attack - XSS | 369 | yes | 0.000 ± 0.000 |

### Per-attack-type recall: Random forest (non-deep, non-private, reference)

| Attack type | Test windows | Unseen in training? | Recall |
|---|---|---|---|
| Bot | 984 | yes | 0.004 ± 0.001 |
| DDoS | 2547 | no | 0.998 ± 0.000 |
| DoS GoldenEye | 6608 | yes | 0.517 ± 0.002 |
| DoS Hulk | 4114 | no | 0.992 ± 0.000 |
| DoS Slowhttptest | 372 | yes | 0.903 ± 0.042 |
| DoS slowloris | 821 | yes | 0.382 ± 0.007 |
| FTP-Patator | 656 | no | 0.646 ± 0.011 |
| Heartbleed | 5 | yes | 0.720 ± 0.228 |
| Infiltration | 35 | yes | 0.011 ± 0.016 |
| PortScan | 3250 | no | 0.986 ± 0.000 |
| SSH-Patator | 1313 | no | 0.552 ± 0.004 |
| Web Attack - Brute Force | 954 | yes | 0.019 ± 0.002 |
| Web Attack - Sql Injection | 20 | yes | 0.000 ± 0.000 |
| Web Attack - XSS | 369 | yes | 0.011 ± 0.003 |

### Operating points (threshold = percentile of benign validation scores)

| Method | Percentile | Recall | FPR | F1 |
|---|---|---|---|---|
| Random forest (non-deep, non-private, reference) | 95 | 0.7290 ± 0.0010 | 0.0558 ± 0.0006 | 0.7947 ± 0.0009 |
| Random forest (non-deep, non-private, reference) | 99 | 0.6923 ± 0.0012 | 0.0144 ± 0.0010 | 0.8052 ± 0.0003 |
| Random forest (non-deep, non-private, reference) | 99.5 | 0.6835 ± 0.0014 | 0.0081 ± 0.0004 | 0.8047 ± 0.0008 |
| Random forest (non-deep, non-private, reference) | 99.9 | 0.6606 ± 0.0035 | 0.0032 ± 0.0003 | 0.7927 ± 0.0023 |
| Centralised hybrid model (non-private) | 95 | 0.6174 ± 0.0115 | 0.0377 ± 0.0018 | 0.7311 ± 0.0086 |
| Centralised hybrid model (non-private) | 99 | 0.5334 ± 0.0071 | 0.0029 ± 0.0005 | 0.6932 ± 0.0059 |
| Centralised hybrid model (non-private) | 99.5 | 0.5180 ± 0.0066 | 0.0021 ± 0.0004 | 0.6807 ± 0.0056 |
| Centralised hybrid model (non-private) | 99.9 | 0.4988 ± 0.0043 | 0.0010 ± 0.0002 | 0.6648 ± 0.0040 |
| Centralised without KG (ablation) | 95 | 0.6370 ± 0.0048 | 0.0337 ± 0.0011 | 0.7491 ± 0.0029 |
| Centralised without KG (ablation) | 99 | 0.5355 ± 0.0080 | 0.0025 ± 0.0004 | 0.6953 ± 0.0067 |
| Centralised without KG (ablation) | 99.5 | 0.5209 ± 0.0070 | 0.0017 ± 0.0002 | 0.6835 ± 0.0061 |
| Centralised without KG (ablation) | 99.9 | 0.5036 ± 0.0067 | 0.0007 ± 0.0002 | 0.6693 ± 0.0058 |
| FedAvg without DP | 95 | 0.6097 ± 0.0083 | 0.0388 ± 0.0021 | 0.7245 ± 0.0056 |
| FedAvg without DP | 99 | 0.5337 ± 0.0083 | 0.0059 ± 0.0008 | 0.6909 ± 0.0070 |
| FedAvg without DP | 99.5 | 0.5177 ± 0.0085 | 0.0045 ± 0.0006 | 0.6784 ± 0.0072 |
| FedAvg without DP | 99.9 | 0.4871 ± 0.0087 | 0.0010 ± 0.0002 | 0.6542 ± 0.0078 |
| Proposed: DP-FL with KG | 95 | 0.5012 ± 0.0036 | 0.0289 ± 0.0010 | 0.6443 ± 0.0031 |
| Proposed: DP-FL with KG | 99 | 0.4218 ± 0.0025 | 0.0015 ± 0.0001 | 0.5922 ± 0.0025 |
| Proposed: DP-FL with KG | 99.5 | 0.4009 ± 0.0040 | 0.0007 ± 0.0001 | 0.5718 ± 0.0040 |
| Proposed: DP-FL with KG | 99.9 | 0.3679 ± 0.0109 | 0.0001 ± 0.0000 | 0.5377 ± 0.0116 |
| DP-FL without KG (ablation) | 95 | 0.4729 ± 0.0062 | 0.0284 ± 0.0022 | 0.6195 ± 0.0056 |
| DP-FL without KG (ablation) | 99 | 0.3978 ± 0.0088 | 0.0017 ± 0.0001 | 0.5678 ± 0.0089 |
| DP-FL without KG (ablation) | 99.5 | 0.3750 ± 0.0086 | 0.0009 ± 0.0002 | 0.5448 ± 0.0090 |
| DP-FL without KG (ablation) | 99.9 | 0.3553 ± 0.0092 | 0.0002 ± 0.0001 | 0.5241 ± 0.0100 |

### Table 4.5 Privacy budget (delta = 1e-6, worst tenant)

| Method | sigma | Tenants per round (fraction) | Rounds of worst tenant | epsilon (Opacus) | epsilon (numpy check) |
|---|---|---|---|---|---|
| Proposed: DP-FL with KG | 1.455 ± 0.000 | 0.30 ± 0.00 | 20.2 ± 2.3 | 6.64 ± 0.41 | 6.65 ± 0.43 |
| DP-FL without KG (ablation) | 1.455 ± 0.000 | 0.30 ± 0.00 | 19.4 ± 1.3 | 6.50 ± 0.25 | 6.50 ± 0.25 |

### Table 4.7 Efficiency

| Method | Latency (ms/window) | Model size (MB) | Communication per tenant per round, up+down (MB) | Training time (hours) |
|---|---|---|---|---|
| Centralised hybrid model (non-private) | 1.55 ± 0.02 | 3.511 ± 0.000 | - | 0.058 ± 0.008 |
| Centralised without KG (ablation) | 1.57 ± 0.11 | 3.511 ± 0.000 | - | 0.054 ± 0.003 |
| FedAvg without DP | 1.53 ± 0.05 | 3.511 ± 0.000 | 7.021 ± 0.000 | 0.060 ± 0.001 |
| Proposed: DP-FL with KG | 3.94 ± 0.10 | 3.511 ± 0.000 | 7.021 ± 0.000 | 0.408 ± 0.001 |
| DP-FL without KG (ablation) | 3.98 ± 0.06 | 3.511 ± 0.000 | 7.021 ± 0.000 | 0.408 ± 0.001 |

### Membership inference (threshold attack; 0.5 = no leakage)

| Model | n | Attack AUC | Attack balanced accuracy |
|---|---|---|---|
| Centralised hybrid model (non-private) | 5 | 0.541 ± 0.003 | 0.498 ± 0.010 |
| Proposed: DP-FL with KG | 5 | 0.546 ± 0.003 | 0.504 ± 0.015 |

### Table 4.6 Explanation fidelity (removal of the top-5 flow features)

| Model | n | Fidelity ratio (SHAP/random) | Alerts flipped (SHAP top-5) | Alerts flipped (random 5) | Spearman vs reference |
|---|---|---|---|---|---|
| Centralised hybrid model (non-private) | 5 | 16.16 ± 2.32 | 0.600 ± 0.030 | 0.042 ± 0.010 | - |
| Proposed: DP-FL with KG | 5 | 11.55 ± 2.27 | 0.506 ± 0.103 | 0.079 ± 0.021 | 0.961 ± 0.006 |
