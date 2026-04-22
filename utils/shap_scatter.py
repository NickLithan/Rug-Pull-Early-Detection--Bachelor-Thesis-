import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import shap
import copy
from sklearn.preprocessing import QuantileTransformer
from typing import Any


def _is_signed(feature: str) -> bool:
    return (
        feature.startswith("roll") or
        feature.startswith("corwin") or
        feature == "ofi"
    )


def _make_binned_ci_bands(sv: Any, n_bins: int, alpha: float) -> tuple[np.ndarray, ...]:
    bin_left_edges = np.arange(n_bins) / n_bins
    bin_centers = bin_left_edges + 1/(2*n_bins)

    mean_line = np.empty(n_bins, dtype=np.float64)
    q_lo = np.empty(n_bins, dtype=np.float64)
    q_high = np.empty(n_bins, dtype=np.float64)

    for i in range(n_bins):
        low, high = bin_left_edges[i], bin_left_edges[i] + 1/n_bins
        # right edge is included only in last bin
        mask = (sv.data >= low) & (sv.data < high + (i == n_bins-1))
        selected_val = sv.values[mask]
        mean_line[i] = np.mean(selected_val)
        q_lo[i], q_high[i] = np.quantile(selected_val, (alpha/2, 1 - alpha/2))
    
    return bin_centers, mean_line, q_lo, q_high


def make_shap_scatter_plot(model: Any, X_test: pd.DataFrame, features: list[str],
                           approximate: bool=False, n_cols: int=3,
                           with_ci_bands: bool=True, n_bins: int=10, alpha: float=0.05):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test, approximate=approximate)

    if approximate:
        shap_values_transformed = copy.deepcopy(shap_values[:, :, 1])
    else:
        shap_values_transformed = copy.deepcopy(shap_values)    

    # dimensions
    n_rows = len(features) // n_cols + 1
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, 9), dpi=250)

    for i, ax_i in enumerate(ax.flatten()):
        if i >= len(features):
            fig.delaxes(ax_i)
            continue

        feature = features[i]
        sv = shap_values_transformed[:, feature]
        
        # quantile transform + scatter
        transform = QuantileTransformer()
        sv.data = transform.fit_transform(sv.data.reshape(-1, 1)).reshape(-1)
        shap.plots.scatter(sv, ax=ax_i, show=False)

        # zero lines
        ax_i.axhline(0, color="lightgray", linestyle="--")
        zero_transformed = transform.transform(np.array(0).reshape(-1, 1)).item()
        if _is_signed(feature):
            ax_i.axvline(zero_transformed, color="orange", label="x=0", linestyle="--")
            ax_i.legend()

        # binned mean + CI
        if with_ci_bands:
            bin_centers, mean_line, q_lo, q_high = _make_binned_ci_bands(sv, n_bins, alpha)
            ax_i.plot(bin_centers, mean_line, color="pink", alpha=0.9)
            # ax_i.plot(bin_centers, q_lo, color="pink", alpha=0.8)
            # ax_i.plot(bin_centers, q_high, color="pink", alpha=0.8)
            # ax_i.fill_between(bin_centers, q_lo, q_high, color="pink", alpha=0.5)

        # Relabel ticks back to original scale; save transform func to remove reference (&QuantileTransformer)
        ax_i.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _, tr=transform.inverse_transform: f"{tr(np.array(x).reshape(-1, 1)).item():.2g}")
        )
        ax_i.xaxis.set_minor_locator(mticker.NullLocator())

    plt.tight_layout()
    plt.show()
