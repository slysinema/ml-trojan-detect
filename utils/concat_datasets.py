import pandas as pd
from read_dataset import ReadDatasets

#read all datasets
data1 = ReadDatasets("../datasets/Benign-Monday-no-metadata.parquet").read_data()
data2 = ReadDatasets("../datasets/Botnet-Friday-no-metadata.parquet").read_data()
data3 = ReadDatasets("../datasets/Infiltration-Thursday-no-metadata.parquet").read_data()

#info about Bening dataset
print("./datasets/Benign-Monday-no-metadata.parquet")
print(data1.info())
print(data1.head())
print("\n")

#info about Botnets dataset
print("./datasets/Botnet-Friday-no-metadata.parquet")
print(data2.info())
print(data2.head())
print("\n")

#info about Infiltration dataset
print("./datasets/Infiltration-Thursday-no-metadata.parquet")
print(data3.info())
print(data3.head())

#concat all datasets in one file result.parquet for ml
df = pd.concat([data1, data2, data3], ignore_index=True)
df.to_parquet("../datasets/result.parquet")

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

result = ReadDatasets("../datasets/result.parquet").read_data()
result = result[columns_to_keep]
result["y"] = (result["Label"] != "Benign").astype(int)

print("result.parquet")
print(result.info())
print(result.head())

