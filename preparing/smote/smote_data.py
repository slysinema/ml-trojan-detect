import pandas as pd
from imblearn.over_sampling import SMOTE

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    smote = SMOTE(random_state=11)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res