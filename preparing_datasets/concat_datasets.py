import pandas as pd
import numpy as np
from preparing_datasets.read_dataset import ReadDataset

def concat_datasets() -> pd.DataFrame:
    data1 = ReadDataset("../datasets/raw/Benign-Monday-no-metadata.parquet").read_data()
    data2 = ReadDataset("../datasets/raw/Botnet-Friday-no-metadata.parquet").read_data()
    data3 = ReadDataset("../datasets/raw/Infiltration-Thursday-no-metadata.parquet").read_data()

    df = pd.concat([data1, data2, data3], ignore_index=True)

    columns_to_keep = [
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Total Fwd Packets", "Total Backward Packets",
        "Fwd Packets Length Total", "Bwd Packets Length Total",
        "Packet Length Min", "Packet Length Max", "Packet Length Mean",
        "Packet Length Std", "Packet Length Variance",
        "Flow Bytes/s", "Flow Packets/s",
        "Label"
    ]

    df = df[columns_to_keep]
    df["y"] = (df["Label"] != "Benign").astype(int)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[(df[numeric_cols] >= 0).all(axis=1)]

    return df