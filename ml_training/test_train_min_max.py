"""
Model Training and Evaluation Script.

This script trains, evaluates, and compares two machine learning models
(Random Forest and LightGBM) for network Trojan detection.
It utilizes datasets that have been preprocessed, Min-Max normalized,
and balanced using SMOTE.

To address the high False Positive rate (Precision-Recall trade-off)
caused by synthetic minority over-sampling, the script evaluates the models
using both standard (0.5) and strict (0.9) classification thresholds.
Confusion matrices are automatically generated and saved for visual analysis.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from ml_training.train_and_evaluate_model import train_and_evaluate_model

def main():
    """
    Main execution pipeline for loading data, training models,
    and generating evaluation plots for Min-Max normalized data.
    """
    print("[INFO] Loading Min-Max prepared datasets into memory...")

    X_train = pd.read_parquet("../datasets/processed/min_max/X_train.parquet")
    X_test = pd.read_parquet("../datasets/processed/min_max/X_test.parquet")
    y_train = pd.read_parquet("../datasets/processed/min_max/y_train.parquet")
    y_test = pd.read_parquet("../datasets/processed/labels/y_test.parquet")

    graphics_save_dir = "../graphics/train/min_max"

    print(f"[INFO] Data successfully loaded! Graphs will be saved to: {graphics_save_dir}\n")
    print("=" * 60)

    # Initialize models
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    lgbm_model = LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    # Train and evaluate Random Forest
    train_and_evaluate_model("Random Forest", rf_model, X_train, X_test, y_train, y_test, threshold=0.5,
                             save_dir=graphics_save_dir)
    train_and_evaluate_model("Random Forest", rf_model, X_train, X_test, y_train, y_test, threshold=0.90,
                             save_dir=graphics_save_dir)

    print("\n" + "=" * 60 + "\n")

    # Train and evaluate LightGBM
    train_and_evaluate_model("LightGBM", lgbm_model, X_train, X_test, y_train, y_test, threshold=0.50,
                             save_dir=graphics_save_dir)
    train_and_evaluate_model("LightGBM", lgbm_model, X_train, X_test, y_train, y_test, threshold=0.90,
                             save_dir=graphics_save_dir)

    print("[SUCCESS] All computations completed successfully!")


if __name__ == "__main__":
    main()