"""
Model Training and Evaluation Script.

This script trains, evaluates, and compares five machine learning models
(Random Forest, LightGBM, XGBoost, Extra Trees, and KNN) for network
Trojan detection.
It utilizes datasets that have been preprocessed, Z-score normalized,
and balanced using SMOTE.

To address the high False Positive rate (Precision-Recall trade-off)
caused by synthetic minority over-sampling, the script evaluates the models
using both standard (0.5) and strict (0.9) classification thresholds.
Confusion matrices are automatically generated and saved for visual analysis.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from preparing_data.ml_training.train_and_evaluate_model import train_and_evaluate_model


def main():
    """
    Main execution pipeline for loading data, training models,
    and generating evaluation plots.
    """
    print("[INFO] Loading prepared datasets into memory...")

    X_train = pd.read_parquet("../datasets/processed/z_score/X_train.parquet")
    X_test = pd.read_parquet("../datasets/processed/z_score/X_test.parquet")
    y_train = pd.read_parquet("../datasets/processed/z_score/y_train.parquet")
    y_test = pd.read_parquet("../datasets/processed/labels/y_test.parquet")

    graphics_save_dir = "../graphics/train"
    models_save_dir = "trained_models/"

    print(f"[INFO] Data successfully loaded! Graphs will be saved to: {graphics_save_dir}\n")
    print("=" * 60)

    # ===================== 1. Random Forest =====================
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    train_and_evaluate_model("Random Forest", rf_model, X_train, X_test, y_train, y_test, threshold=0.5,
                             save_dir=graphics_save_dir, model_save_dir=models_save_dir)
    train_and_evaluate_model("Random Forest", rf_model, X_train, X_test, y_train, y_test, threshold=0.90,
                             save_dir=graphics_save_dir)

    print("\n" + "=" * 60 + "\n")

    # ===================== 2. LightGBM =====================
    lgbm_model = LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=42, verbosity=-1)

    train_and_evaluate_model("LightGBM", lgbm_model, X_train, X_test, y_train, y_test, threshold=0.50,
                             save_dir=graphics_save_dir, model_save_dir=models_save_dir)
    train_and_evaluate_model("LightGBM", lgbm_model, X_train, X_test, y_train, y_test, threshold=0.90,
                             save_dir=graphics_save_dir)

    print("\n" + "=" * 60 + "\n")

    # ===================== 3. XGBoost =====================
    xgb_model = XGBClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    train_and_evaluate_model("XGBoost", xgb_model, X_train, X_test, y_train, y_test, threshold=0.50,
                             save_dir=graphics_save_dir, model_save_dir=models_save_dir)
    train_and_evaluate_model("XGBoost", xgb_model, X_train, X_test, y_train, y_test, threshold=0.90,
                             save_dir=graphics_save_dir)

    print("\n" + "=" * 60 + "\n")

    # ===================== 4. Extra Trees =====================
    et_model = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    train_and_evaluate_model("Extra Trees", et_model, X_train, X_test, y_train, y_test, threshold=0.50,
                             save_dir=graphics_save_dir, model_save_dir=models_save_dir)
    train_and_evaluate_model("Extra Trees", et_model, X_train, X_test, y_train, y_test, threshold=0.90,
                             save_dir=graphics_save_dir)

    print("\n" + "=" * 60 + "\n")

    # ===================== 5. KNN =====================
    knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    train_and_evaluate_model("KNN", knn_model, X_train, X_test, y_train, y_test, threshold=0.50,
                             save_dir=graphics_save_dir, model_save_dir=models_save_dir)
    train_and_evaluate_model("KNN", knn_model, X_train, X_test, y_train, y_test, threshold=0.90,
                             save_dir=graphics_save_dir)

    print("\n" + "=" * 60)
    print("[SUCCESS] All 5 models trained and evaluated successfully!")


if __name__ == "__main__":
    main()