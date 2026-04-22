import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from utils.delong import delong_roc_test
from typing import Callable
from tqdm.auto import tqdm


def paired_bootstrap_p_value(y_true: pd.Series|np.ndarray,
                             prediction1: pd.Series|np.ndarray, prediction2: pd.Series|np.ndarray,
                             metric_fn: Callable, seed, n_bootstrap=10_000):
    """Classic boostrap estimate of p-value of change in the metric."""

    rng = np.random.RandomState(seed) # np.random.seed() is legacy
    y_true_arr = y_true.values if isinstance(y_true, pd.Series) else y_true
    pred1_arr = prediction1.values if isinstance(prediction1, pd.Series) else prediction1
    pred2_arr = prediction2.values if isinstance(prediction2, pd.Series) else prediction2
    n = len(y_true)

    observed_diff = metric_fn(y_true_arr, pred1_arr) - metric_fn(y_true_arr, pred2_arr)

    # bootstrap sampling
    indices = rng.randint(0, n, size=(n_bootstrap, n))

    diffs = np.empty(n_bootstrap)
    # for i, idx in tqdm(enumerate(indices), total=n_bootstrap): # debug
    for i, idx in enumerate(indices):
        diffs[i] = (metric_fn(y_true_arr[idx], pred1_arr[idx])
                    - metric_fn(y_true_arr[idx], pred2_arr[idx]))

    p_value = np.mean(np.abs(diffs - observed_diff) >= np.abs(observed_diff))
    return p_value


def paired_permutation_p_value(y_true: pd.Series|np.ndarray,
                               prediction1: pd.Series|np.ndarray, prediction2: pd.Series|np.ndarray,
                               metric_fn: Callable, seed, n_permutations=10_000):
    """Permutation test p-value of change in the metric."""

    rng = np.random.default_rng(seed)
    y_true_arr = y_true.values if isinstance(y_true, pd.Series) else y_true
    pred1_arr = prediction1.values if isinstance(prediction1, pd.Series) else prediction1
    pred2_arr = prediction2.values if isinstance(prediction2, pd.Series) else prediction2

    observed_diff = metric_fn(y_true_arr, pred1_arr) - metric_fn(y_true_arr, pred2_arr)

    duplicated_pred_stack = np.tile(np.vstack((pred1_arr, pred2_arr)), (n_permutations, 1, 1))
    exchanged_pred_samples = rng.permuted(duplicated_pred_stack, axis=1) # generate all permutations at once

    diffs_null = np.empty(n_permutations)
    for i, (null1, null2) in tqdm(enumerate(exchanged_pred_samples), total=n_permutations):
        diffs_null[i] = (metric_fn(y_true_arr, null1) - metric_fn(y_true_arr, null2))

    p_value = np.mean(np.abs(diffs_null) >= np.abs(observed_diff))
    return p_value


def print_metrics(y_true: pd.Series|np.ndarray, prediction: pd.Series|np.ndarray):
    print(f"\tROC AUC\t= {roc_auc_score(y_true, prediction):.4f}")
    print(f"\tPR AUC\t= {average_precision_score(y_true, prediction):.4f}")
    print(f"\tF1\t= {f1_score(y_true, np.rint(prediction)):.4f}")


def compare_models(y_true: pd.Series|np.ndarray,
                   prediction1: pd.Series|np.ndarray, prediction2: pd.Series|np.ndarray,
                   seed: int):
    print("p-value of difference in:")
    print("\tROC AUC\t=", 
          (10 ** delong_roc_test(y_true, prediction1, prediction2)).item())
    print("\tPR AUC\t= {:.4f}".format(
        paired_bootstrap_p_value(y_true, prediction1, prediction2,
        # paired_permutation_p_value(y_true, prediction1, prediction2,
                                 average_precision_score, seed).item()))
    print("\tF1\t= {:.4f}".format(
        paired_bootstrap_p_value(y_true, np.rint(prediction1), np.rint(prediction2),
        # paired_permutation_p_value(y_true, np.rint(prediction1), np.rint(prediction2),
                                 f1_score, seed).item()))
