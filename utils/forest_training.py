import optuna
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


def rf_train_default(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                 output_file: str, seed: int, force_retrain: bool=False, save: bool=True):
    """Trains catboost clf with default params."""

    file = f"model_storage/random_forest/default/{output_file}.cbm"
    if Path(file).is_file() and not force_retrain:
        print("Already trained.")
        model = joblib.load(file)
        pred_proba = model.predict_proba(X_test)[:, 1]
        return model, pred_proba

    model = RandomForestClassifier(random_state=seed)
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)[:,1]

    if save:
        joblib.dump(model, file)

    return model, pred_proba
