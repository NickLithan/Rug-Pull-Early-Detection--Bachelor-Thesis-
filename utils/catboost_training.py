"""Based on https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_pruning.py."""

import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


def cb_train_default(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                 output_file: str, seed: int, force_retrain: bool=False, save: bool=True):
    """Trains catboost clf with default params."""

    file = f"model_storage/catboost/default/{output_file}.cbm"
    if Path(file).is_file() and not force_retrain:
        print("Already trained.")
        model = CatBoostClassifier()
        model.load_model(file)
        pred_proba = model.predict_proba(X_test)[:, 1]
        return model, pred_proba

    model = CatBoostClassifier(verbose=0, random_seed=seed)
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)[:,1]
    
    if save:
        model.save_model(file)

    return model, pred_proba
    


def make_standard_objective(X_train: pd.DataFrame, y_train: pd.Series, seed: int, n_splits: int=5):
    """Makes an Optuna objective function: walk-forward cross-validation ROC AUC for catboost."""

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1500),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "max_bin": trial.suggest_int("max_bin", 32, 512),
            "random_strength": trial.suggest_float("random_strength", 1e-2, 10.0, log=True),
            "verbose": 0, "random_seed": seed,
            "auto_class_weights": "Balanced",
        }

        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = CatBoostClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=0,
            )

            preds = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, preds))

            # stop unpromising trials early
            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    return objective


def cb_tune_and_train_standard(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                            output_file: str, seed: int, n_trials: int=25, force_retrain: bool=False, save: bool = True):
    """Tune catboost hyperparameters with Optuna + walk-forward CV, then retrain on full training set."""

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    file = f"model_storage/catboost/tuned_standard/{output_file}.cbm"
    if Path(file).is_file() and not force_retrain:
        print("Already trained.")
        model = CatBoostClassifier()
        model.load_model(file)
        pred_proba = model.predict_proba(X_test)[:, 1]
        return model, pred_proba

    objective = make_standard_objective(X_train, y_train, seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best CV ROC AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # retraining:
    best_params = study.best_params
    best_params["verbose"], best_params["random_seed"] = 0, seed

    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)[:, 1]
    
    if save:
        model.save_model(file)

    return model, pred_proba
