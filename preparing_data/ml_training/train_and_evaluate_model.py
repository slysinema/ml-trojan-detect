import os
import time

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def train_and_evaluate_model(model_name: str, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_train: pd.DataFrame, y_test: pd.DataFrame, threshold: float = 0.5,
                             save_dir: str = None, model_save_dir: str = None):
    """
    Trains a machine learning model, evaluates its performance with a custom threshold,
    and generates a confusion matrix heatmap.

    This function is designed to handle the Precision-Recall trade-off, which is
    crucial after applying techniques like SMOTE. By adjusting the classification
    threshold, it allows for fine-tuning the model's sensitivity. It outputs the
    training time, a detailed classification report, and saves a visual representation
    of the confusion matrix.

    Args:
        model_name (str): The display name of the algorithm (e.g., "LightGBM").
        model: The machine learning model instance (must support fit and predict_proba).
        X_train (pd.DataFrame): The training dataset features.
        X_test (pd.DataFrame): The testing dataset features.
        y_train (pd.DataFrame): The training dataset labels.
        y_test (pd.DataFrame): The testing dataset labels.
        threshold (float): The probability threshold for the positive class. Defaults to 0.5.
        save_dir (str): The directory to save the confusion matrix plot. If None, shows the plot.
    """
    print(f"[START] - Training model {model_name} (Threshold: {threshold})...")

    start_time = time.time()

    # Train the model
    model.fit(X_train, y_train.values.ravel())

    # Save train model
    if model_save_dir:
        os.makedirs(model_save_dir, exist_ok=True)
        clean_name = model_name.lower().replace(" ", "_")
        model_path = f"{model_save_dir}/{clean_name}.joblib"

        # Сохраняем модель на диск
        joblib.dump(model, model_path)
        print(f"[MODEL] - Trained model saved to: {model_path}")

    # Get probabilities for the positive class (Trojan = 1) and apply the custom threshold
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    end_time = time.time()

    print(f"[STOP] - Training time for {model_name}: {end_time - start_time:.2f} seconds")
    print(f"\n[REPORT] {model_name} (Threshold: {threshold}):\n{classification_report(y_test, y_pred)}")

    # Calculate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Generate a beautiful heatmap for the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Benign (0)', 'Trojan (1)'],
                yticklabels=['Benign (0)', 'Trojan (1)'])

    plt.title(f'Confusion Matrix: {model_name}\n(Threshold: {threshold})', fontsize=14)
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)

    # Save or show the plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        clean_name = model_name.lower().replace(" ", "_")
        file_path = f"{save_dir}/cm_{clean_name}_{str(threshold).replace('.', '_')}.png"

        plt.savefig(file_path, bbox_inches='tight')
        print(f"[GRAPH] - Graph successfully saved to: {file_path}")
    else:
        plt.show()

    plt.close()