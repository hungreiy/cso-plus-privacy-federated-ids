"""CSE-CIC-IDS2018 loader (Processed Traffic Data for ML Algorithms, CICFlowMeter-V3 CSVs).

The 2018 files use abbreviated column names. Every feature is mapped to the CIC-IDS2017 MachineLearningCVE name and
put in the 2017 order, so the same model, scaler and knowledge-graph lookup work on both datasets and a model trained
on one dataset can be tested on the other. The 2017 release has a duplicated column 'Fwd Header Length.1'; for 2018
it is filled with 'Fwd Header Len'. Timestamp, Protocol and the extra identifier columns of some days (Flow ID, Src IP,
Src Port, Dst IP) are dropped; flows are sorted by Timestamp within each file; header lines repeated inside a file are
removed.
"""
import numpy as np, pandas as pd

ORDER_2017 = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets',
 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length',
 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count',
 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
 'Idle Max', 'Idle Min']

MAP_2018 = {'Dst Port': 'Destination Port', 'Flow Duration': 'Flow Duration', 'Tot Fwd Pkts': 'Total Fwd Packets',
 'Tot Bwd Pkts': 'Total Backward Packets', 'TotLen Fwd Pkts': 'Total Length of Fwd Packets', 'TotLen Bwd Pkts': 'Total Length of Bwd Packets',
 'Fwd Pkt Len Max': 'Fwd Packet Length Max', 'Fwd Pkt Len Min': 'Fwd Packet Length Min', 'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
 'Fwd Pkt Len Std': 'Fwd Packet Length Std', 'Bwd Pkt Len Max': 'Bwd Packet Length Max', 'Bwd Pkt Len Min': 'Bwd Packet Length Min',
 'Bwd Pkt Len Mean': 'Bwd Packet Length Mean', 'Bwd Pkt Len Std': 'Bwd Packet Length Std', 'Flow Byts/s': 'Flow Bytes/s',
 'Flow Pkts/s': 'Flow Packets/s', 'Flow IAT Mean': 'Flow IAT Mean', 'Flow IAT Std': 'Flow IAT Std', 'Flow IAT Max': 'Flow IAT Max',
 'Flow IAT Min': 'Flow IAT Min', 'Fwd IAT Tot': 'Fwd IAT Total', 'Fwd IAT Mean': 'Fwd IAT Mean', 'Fwd IAT Std': 'Fwd IAT Std',
 'Fwd IAT Max': 'Fwd IAT Max', 'Fwd IAT Min': 'Fwd IAT Min', 'Bwd IAT Tot': 'Bwd IAT Total', 'Bwd IAT Mean': 'Bwd IAT Mean',
 'Bwd IAT Std': 'Bwd IAT Std', 'Bwd IAT Max': 'Bwd IAT Max', 'Bwd IAT Min': 'Bwd IAT Min', 'Fwd PSH Flags': 'Fwd PSH Flags',
 'Bwd PSH Flags': 'Bwd PSH Flags', 'Fwd URG Flags': 'Fwd URG Flags', 'Bwd URG Flags': 'Bwd URG Flags', 'Fwd Header Len': 'Fwd Header Length',
 'Bwd Header Len': 'Bwd Header Length', 'Fwd Pkts/s': 'Fwd Packets/s', 'Bwd Pkts/s': 'Bwd Packets/s', 'Pkt Len Min': 'Min Packet Length',
 'Pkt Len Max': 'Max Packet Length', 'Pkt Len Mean': 'Packet Length Mean', 'Pkt Len Std': 'Packet Length Std', 'Pkt Len Var': 'Packet Length Variance',
 'FIN Flag Cnt': 'FIN Flag Count', 'SYN Flag Cnt': 'SYN Flag Count', 'RST Flag Cnt': 'RST Flag Count', 'PSH Flag Cnt': 'PSH Flag Count',
 'ACK Flag Cnt': 'ACK Flag Count', 'URG Flag Cnt': 'URG Flag Count', 'CWE Flag Count': 'CWE Flag Count', 'ECE Flag Cnt': 'ECE Flag Count',
 'Down/Up Ratio': 'Down/Up Ratio', 'Pkt Size Avg': 'Average Packet Size', 'Fwd Seg Size Avg': 'Avg Fwd Segment Size',
 'Bwd Seg Size Avg': 'Avg Bwd Segment Size', 'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk', 'Fwd Pkts/b Avg': 'Fwd Avg Packets/Bulk',
 'Fwd Blk Rate Avg': 'Fwd Avg Bulk Rate', 'Bwd Byts/b Avg': 'Bwd Avg Bytes/Bulk', 'Bwd Pkts/b Avg': 'Bwd Avg Packets/Bulk',
 'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate', 'Subflow Fwd Pkts': 'Subflow Fwd Packets', 'Subflow Fwd Byts': 'Subflow Fwd Bytes',
 'Subflow Bwd Pkts': 'Subflow Bwd Packets', 'Subflow Bwd Byts': 'Subflow Bwd Bytes', 'Init Fwd Win Byts': 'Init_Win_bytes_forward',
 'Init Bwd Win Byts': 'Init_Win_bytes_backward', 'Fwd Act Data Pkts': 'act_data_pkt_fwd', 'Fwd Seg Size Min': 'min_seg_size_forward',
 'Active Mean': 'Active Mean', 'Active Std': 'Active Std', 'Active Max': 'Active Max', 'Active Min': 'Active Min',
 'Idle Mean': 'Idle Mean', 'Idle Std': 'Idle Std', 'Idle Max': 'Idle Max', 'Idle Min': 'Idle Min'}
DROP_2018 = {'Timestamp', 'Protocol', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP'}
HOLDOUT_2018 = 'DoS attacks-GoldenEye,DoS attacks-Slowloris,DoS attacks-SlowHTTPTest,Bot,Brute Force -Web,Brute Force -XSS,SQL Injection,Infilteration'

def check_columns(path):
    """Return (unmapped columns, missing 2017 features) for one 2018 file, reading only the header."""
    cols = [c.strip() for c in pd.read_csv(path, nrows=0).columns]
    unmapped = [c for c in cols if c not in MAP_2018 and c not in DROP_2018 and c != 'Label']
    got = {MAP_2018[c] for c in cols if c in MAP_2018}
    missing = [c for c in ORDER_2017 if c not in got and c != 'Fwd Header Length.1']
    return unmapped, missing

def read_csv_2018(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df[df['Label'].astype(str).str.strip() != 'Label']            # header lines repeated inside some files
    if 'Timestamp' in df.columns:
        ts = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        df = df.assign(_ts=ts).sort_values('_ts', kind='stable').drop(columns=['_ts'])
    unmapped = [c for c in df.columns if c not in MAP_2018 and c not in DROP_2018 and c != 'Label']
    if unmapped: raise SystemExit(f'{path}: unknown columns {unmapped} - add them to MAP_2018 or DROP_2018 in ids2018.py')
    labels = df['Label'].astype(str).str.strip().to_numpy()
    feats = df[[c for c in df.columns if c in MAP_2018]].rename(columns=MAP_2018)
    feats['Fwd Header Length.1'] = feats['Fwd Header Length']
    feats = feats[ORDER_2017].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    ok = feats.notna().all(axis=1).to_numpy()
    return feats.to_numpy(dtype=np.float32)[ok], labels[ok], list(ORDER_2017), int((~ok).sum())
