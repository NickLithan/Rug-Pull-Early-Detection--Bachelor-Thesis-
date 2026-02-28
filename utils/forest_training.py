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

    file = f"model_storage/random_forest/default/{output_file}.joblib"
    if Path(file).is_file() and not force_retrain:
        print("Already trained.")
        model = joblib.load(file)
        pred_proba = model.predict_proba(X_test)[:, 1]
        return model, pred_proba

    model = RandomForestClassifier(random_state=seed)
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)[:,1]

    if save:
        joblib.dump(model, file, compress=("zlib", 3))

    return model, pred_proba


def make_rf_objective(X_train: pd.DataFrame, y_train: pd.Series, seed: int, n_splits: int = 5):
    """Makes an Optuna objective function: walk-forward cross-validation ROC AUC for RF."""

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 2000),
            "max_depth":         trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 50),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "max_samples":       trial.suggest_float("max_samples", 0.5, 1.0),
            "criterion":         trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "class_weight":      "balanced",
            "random_state":      seed,
            "n_jobs":            3,
        }

        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = RandomForestClassifier(**params)
            model.fit(X_tr, y_tr)

            preds = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, preds))

            # stop unpromising trials early
            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    return objective


def rf_tune_and_train_standard(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                               output_file: str, seed: int, n_trials: int = 25,
                               force_retrain: bool = False, save: bool = True):
    """Tune RF hyperparameters with Optuna + walk-forward CV, then retrain on full training set."""

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    file = f"model_storage/random_forest/tuned_standard/{output_file}.joblib"
    if Path(file).is_file() and not force_retrain:
        print("Already trained.")
        model = joblib.load(file)
        pred_proba = model.predict_proba(X_test)[:, 1]
        return model, pred_proba

    objective = make_rf_objective(X_train, y_train, seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best CV ROC AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # retrain on full training data
    best_params = study.best_params
    best_params["class_weight"] = "balanced"
    best_params["random_state"] = seed
    best_params["n_jobs"] = 3

    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)[:, 1]

    if save:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, file, compress=("zlib", 3))

    return model, pred_proba
