from sklearn.model_selection import train_test_split
from preparing.read_dataset import ReadDataset

"""
Data Preparation Module

Splits the cleaned dataset into training and testing sets
using an 80/20 ratio with stratification to preserve class balance.

Dataset: full_dataset_after_clean.parquet
    - X: 15 network traffic features
    - y: binary label (0 = Benign, 1 = Attack)

Split:
    - Train: 80%
    - Test:  20%
"""

result = ReadDataset("../datasets/result/full/full_dataset_after_clean.parquet").read_data()

X = result.drop(columns=["Label", "y"])
y = result["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=11,
    shuffle=True,
    stratify=y
)

print(f"Train: {len(X_train)} rows")
X_train.to_parquet("../datasets/result/train_test_split/training.parquet")
print(f"Test:  {len(X_test)} rows")
X_test.to_parquet("../datasets/result/train_test_split/testing.parquet")