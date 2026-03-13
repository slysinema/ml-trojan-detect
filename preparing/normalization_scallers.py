from typing import Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


def min_max_scaler(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Min-Max scaling to the datasets."""
    scaler = MinMaxScaler()
    X_train_minmax = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_minmax = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_minmax, X_test_minmax


def standard_scaler(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Z-score standardization to the datasets."""
    scaler = StandardScaler()
    X_train_z_score = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_z_score = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_z_score, X_test_z_score