import os

from preparing_data.preparing_datasets.concat_datasets import concat_datasets
from preparing_data.preparing_datasets.train_test_split import preparing_train_test_split
from preparing_data.preparing_datasets.normalization_scalers import standard_scaler, min_max_scaler
from preparing_data.preparing_datasets.smote_data import apply_smote

# Step 1: Load and concat datasets
result = concat_datasets()

# Step 2: Split into train/test
X_train, X_test, y_train, y_test = preparing_train_test_split(result)

# Step 3: Normalize (both scalers)
X_train_minmax, X_test_minmax = min_max_scaler(X_train, X_test)
X_train_zscore, X_test_zscore = standard_scaler(X_train, X_test)

# Step 4: Balance classes with SMOTE
X_train_minmax, y_train_minmax = apply_smote(X_train_minmax, y_train)
X_train_zscore, y_train_zscore = apply_smote(X_train_zscore, y_train)

# Step 5: Save
os.makedirs("../datasets/processed/min_max", exist_ok=True)
os.makedirs("../datasets/processed/z_score", exist_ok=True)
os.makedirs("../datasets/processed/labels", exist_ok=True)

X_train_minmax.to_parquet("../datasets/processed/min_max/X_train.parquet")
X_test_minmax.to_parquet("../datasets/processed/min_max/X_test.parquet")

X_train_zscore.to_parquet("../datasets/processed/z_score/X_train.parquet")
X_test_zscore.to_parquet("../datasets/processed/z_score/X_test.parquet")

y_train_minmax.to_frame().to_parquet("../datasets/processed/min_max/y_train.parquet")
y_train_zscore.to_frame().to_parquet("../datasets/processed/z_score/y_train.parquet")
y_test.to_frame().to_parquet("../datasets/processed/labels/y_test.parquet")

print("Done!")