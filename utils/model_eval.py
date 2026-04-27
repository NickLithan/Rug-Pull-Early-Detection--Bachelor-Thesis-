import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from utils.delong import delong_roc_test
from typing import Callable
from tqdm.auto import tqdm, trange


def paired_bootstrap_p_value(y_true: pd.Series|np.ndarray,
                             prediction1: pd.Series|np.ndarray, prediction2: pd.Series|np.ndarray,
                             metric_fn: Callable, seed, n_bootstrap=10_000) -> float:
    """Classic boostrap estimate of p-value of change in the metric."""

    rng = np.random.RandomState(seed) # np.random.seed() is legacy
    y_true_arr = y_true.values if isinstance(y_true, pd.Series) else y_true
    pred1_arr = prediction1.values if isinstance(prediction1, pd.Series) else prediction1
    pred2_arr = prediction2.values if isinstance(prediction2, pd.Series) else prediction2
    n = len(y_true)

    observed_diff = metric_fn(y_true_arr, pred1_arr) - metric_fn(y_true_arr, pred2_arr)

    # bootstrap sampling
    indices = rng.randint(0, n, size=(n_bootstrap, n))

    diffs = np.empty(n_bootstrap, dtype=np.float64)
    # for i, idx in tqdm(enumerate(indices), total=n_bootstrap): # debug
    for i, idx in enumerate(indices):
        diffs[i] = (metric_fn(y_true_arr[idx], pred1_arr[idx])
                    - metric_fn(y_true_arr[idx], pred2_arr[idx]))

    p_value = np.mean(np.abs(diffs - observed_diff) >= np.abs(observed_diff))
    return p_value


def paired_permutation_p_value(y_true: pd.Series|np.ndarray,
                               prediction1: pd.Series|np.ndarray, prediction2: pd.Series|np.ndarray,
                               metric_fn: Callable, seed, n_permutations=10_000) -> float:
    """Permutation test p-value of change in the metric."""

    rng = np.random.default_rng(seed)
    y_true_arr = y_true.values if isinstance(y_true, pd.Series) else y_true
    pred1_arr = prediction1.values if isinstance(prediction1, pd.Series) else prediction1
    pred2_arr = prediction2.values if isinstance(prediction2, pd.Series) else prediction2

    observed_diff = metric_fn(y_true_arr, pred1_arr) - metric_fn(y_true_arr, pred2_arr)

    duplicated_pred_stack = np.tile(np.vstack((pred1_arr, pred2_arr)), (n_permutations, 1, 1))
    exchanged_pred_samples = rng.permuted(duplicated_pred_stack, axis=1) # generate all permutations at once

    diffs_null = np.empty(n_permutations, dtype=np.float64)
    # for i, (null1, null2) in tqdm(enumerate(exchanged_pred_samples), total=n_permutations):
    for i, (null1, null2) in enumerate(exchanged_pred_samples):
        diffs_null[i] = (metric_fn(y_true_arr, null1) - metric_fn(y_true_arr, null2))

    p_value = np.mean(np.abs(diffs_null) >= np.abs(observed_diff))
    return p_value


def paired_blocked_bootstrap_p_value(y_true: pd.Series | np.ndarray,
                                     prediction1: pd.Series | np.ndarray, prediction2: pd.Series | np.ndarray,
                                     metric_fn: Callable, seed: int, block_size: int=100,
                                     n_bootstrap: int=10_000) -> float:
    """Blocked boostrap estimate of p-value of change in the metric (block_size should be selected)."""

    rng = np.random.RandomState(seed)
    y_true_arr = y_true.values if isinstance(y_true, pd.Series) else y_true
    pred1_arr = prediction1.values if isinstance(prediction1, pd.Series) else prediction1
    pred2_arr = prediction2.values if isinstance(prediction2, pd.Series) else prediction2
    n = len(y_true)
    assert block_size <= n, f"block_size ({block_size}) must be <= n ({n})."

    observed_diff = metric_fn(y_true_arr, pred1_arr) - metric_fn(y_true_arr, pred2_arr)

    # a block starting at position s covers [s, s + block_size).
    max_start = n - block_size
    n_blocks = int(np.ceil(n / block_size))

    # bootstrap sample of block starts -> all block idx
    block_starts = rng.randint(0, max_start + 1, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block_size)
    all_indices = (
        block_starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]
    ).reshape(n_bootstrap, -1)[:, :n]

    diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i, idx in enumerate(all_indices):
        diffs[i] = (metric_fn(y_true_arr[idx], pred1_arr[idx])
                    - metric_fn(y_true_arr[idx], pred2_arr[idx]))

    p_value = np.mean(np.abs(diffs - observed_diff) >= np.abs(observed_diff))
    return p_value


def print_metrics(y_true: pd.Series|np.ndarray, prediction: pd.Series|np.ndarray):
    print(f"\tROC AUC\t= {roc_auc_score(y_true, prediction):.4f}")
    print(f"\tPR AUC\t= {average_precision_score(y_true, prediction):.4f}")
    print(f"\tF1\t= {f1_score(y_true, np.rint(prediction)):.4f}")


# def compare_models(y_true: pd.Series|np.ndarray,
#                    prediction1: pd.Series|np.ndarray, prediction2: pd.Series|np.ndarray,
#                    seed: int):
#     print("p-value of difference in:")
#     print("\tROC AUC\t=", 
#           (10 ** delong_roc_test(y_true, prediction1, prediction2)).item())
#     print("\tPR AUC\t= {:.4f}".format(
#         paired_bootstrap_p_value(y_true, prediction1, prediction2,
#         # paired_permutation_p_value(y_true, prediction1, prediction2,
#                                  average_precision_score, seed).item()))
#     print("\tF1\t= {:.4f}".format(
#         paired_bootstrap_p_value(y_true, np.rint(prediction1), np.rint(prediction2),
#         # paired_permutation_p_value(y_true, np.rint(prediction1), np.rint(prediction2),
#                                  f1_score, seed).item()))


# wrappper class
class ModelComparison:
    def __init__(self, y_true: pd.Series|np.ndarray,
                 prediction1: pd.Series|np.ndarray, prediction2: pd.Series|np.ndarray,
                 seed: int):
        self.y = y_true
        self.pred1 = prediction1
        self.pred2 = prediction2
        self.seed = seed

    def _get_metrics(self) -> dict[str, Callable]:
        return {
            "ROC AUC": roc_auc_score,
            "PR AUC": average_precision_score,
            "F1": lambda y, p: f1_score(y, np.rint(p))
        }

    def _metric_to_default_p_value_method(self, metric: str):
        if metric == "ROC AUC":
            return lambda y, pred1, pred2, metric_fn, seed: (10 ** delong_roc_test(y, pred1, pred2)).item()
        return paired_bootstrap_p_value

    def _get_test_fns(self, metric: str) -> dict[str, Callable]:
        return {
            "default": self._metric_to_default_p_value_method(metric),
            "permutation": paired_permutation_p_value,
            "blocked bootstrap": paired_blocked_bootstrap_p_value,
        }

    def compare_models_table(self):
        metric_strs = list(self._get_metrics())
        test_fn_strs = list(self._get_test_fns(metric_strs[0]))

        out_table = pd.DataFrame({
            metric_str: np.empty(len(test_fn_strs)) for metric_str in metric_strs
        }, index=test_fn_strs)

        n_rows, n_cols = len(test_fn_strs), len(metric_strs)
        for i in trange(n_rows * n_cols, desc="Comparing metrics..."):
            row, col = i//3, i%3
            metric_str, test_str = metric_strs[col], test_fn_strs[row]
            metric_fn = self._get_metrics()[metric_str]
            test_fn = self._get_test_fns(metric_str)[test_str]
            out_table.iloc[row, col] = test_fn(
                self.y, self.pred1, self.pred2, metric_fn, self.seed
            )

        return out_table.copy()
