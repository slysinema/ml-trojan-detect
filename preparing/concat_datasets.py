import pandas as pd
from read_dataset import ReadDataset
import numpy as np

#read all datasets
data1 = ReadDataset("../datasets/raw/Benign-Monday-no-metadata.parquet").read_data()
data2 = ReadDataset("../datasets/raw/Botnet-Friday-no-metadata.parquet").read_data()
data3 = ReadDataset("../datasets/raw/Infiltration-Thursday-no-metadata.parquet").read_data()

#info about Bening dataset
print("./datasets/raw/Benign-Monday-no-metadata.parquet")
data1.info()
print(data1.head())
print("\n")

#info about Botnets dataset
print("./datasets/raw/Botnet-Friday-no-metadata.parquet")
data2.info()
print(data2.head())
print("\n")

#info about Infiltration dataset
print("./datasets/raw/Infiltration-Thursday-no-metadata.parquet")
data3.info()
print(data3.head())

#concat all datasets in one file result.parquet for ml
df = pd.concat([data1, data2, data3], ignore_index=True)
df.to_parquet("../datasets/result/full_dataset_before_clean.parquet")

#normalization result dataset based on the analysis from the bachelor's thesis
#leave only 15 columns for further machine training
columns_to_keep = [
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packets Length Total", "Bwd Packets Length Total",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance",
    "Flow Bytes/s", "Flow Packets/s",
    "Label"
]

result = df
result = result[columns_to_keep]
result["y"] = (result["Label"] != "Benign").astype(int)

print("full_dataset_before_clean.parquet")
result.info()
print(result.head())

#cleaning full dataset(deleting rows with Infinity/NaN/Null columns)
result = result.replace([np.inf, -np.inf], np.nan).dropna()
result.to_parquet("../datasets/result/full_dataset_after_clean.parquet")
print("full_dataset_after_clean.parquet")
result.info()

