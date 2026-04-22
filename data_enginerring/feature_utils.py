import numpy as np
import pandas as pd


def _make_time_bars_ohlc(timedeltas_sorted: np.ndarray, midquote_sorted: np.ndarray,
                         bar_t: float, window: float=300) -> dict[str, np.ndarray]:
    """
    Returns OHLC arrays in specified time bars. Empty bars are filled with NaN.
    Allows and manages nulls in midquote series.
    """

    # drop null-price ticks first
    valid = ~np.isnan(midquote_sorted)
    timedeltas_sorted = timedeltas_sorted[valid]
    midquote_sorted = midquote_sorted[valid]

    n_bars = int(window / bar_t)

    # edge case: all null
    if len(midquote_sorted) == 0:
        empty = np.full(n_bars, np.nan)
        return {"open": empty, "high": empty.copy(), "low": empty.copy(),
                "close": empty.copy(), "is_filled": np.zeros(n_bars, dtype=bool)}

    # t -> (t / bar_t) -> idx = floor(..) -> t_change = flatnonzero(..)
    bar_idx = np.floor(timedeltas_sorted / bar_t).astype(int)
    bar_idx = np.clip(bar_idx, 0, n_bars - 1)

    # bar boundaries
    t_change = np.flatnonzero(np.diff(bar_idx)) + 1
    seg_starts = np.concatenate([[0], t_change])
    seg_ends = np.concatenate([t_change - 1, [len(midquote_sorted) - 1]])

    # drop empty bars
    filled_bars = bar_idx[seg_starts]

    # OHLC; "reduceat" allows for lightning speed slices
    open_vals = midquote_sorted[seg_starts]
    close_vals = midquote_sorted[seg_ends]
    high_vals = np.maximum.reduceat(midquote_sorted, seg_starts)
    low_vals = np.minimum.reduceat(midquote_sorted, seg_starts)

    # managing nan
    open_arr = np.full(n_bars, np.nan)
    high_arr = np.full(n_bars, np.nan)
    low_arr = np.full(n_bars, np.nan)
    close_arr = np.full(n_bars, np.nan)

    open_arr[filled_bars] = open_vals
    high_arr[filled_bars] = high_vals
    low_arr[filled_bars] = low_vals
    close_arr[filled_bars] = close_vals

    # mask
    is_filled_arr = np.zeros(n_bars, dtype=bool)
    is_filled_arr[filled_bars] = True

    return {"open": open_arr, "high": high_arr, "low": low_arr, "close": close_arr, "is_filled": is_filled_arr}


def _ffill_ohlc(ohlc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Forward-fills empty OHLC bars with last observed Close."""

    close = ohlc["close"]
    is_filled = ohlc["is_filled"]

    if not np.any(is_filled) or np.all(is_filled):
        return {k: v.copy() for k, v in ohlc.items()}

    # last_obs = roll_max(idx if is_filled else null)
    last_valid_idx = np.where(is_filled, np.arange(len(close)), 0)
    np.maximum.accumulate(last_valid_idx, out=last_valid_idx)
    fill_vals = close[last_valid_idx]

    empty = ~is_filled
    out = {}
    for key in ("open", "high", "low", "close"):
        arr = ohlc[key].copy()
        arr[empty] = fill_vals[empty]
        out[key] = arr

    out["is_filled"] = np.ones(len(is_filled), dtype=bool)
    return out


def series2time_bar_ohlc(t_rel_s: pd.Series, midquote: pd.Series,
                         bar_t: float = 10, window: float = 300,
                         ffill: bool = True) -> dict[str, np.ndarray]:
    """Calculates time bars OHLC from timedeltas and midquote series."""

    timedeltas_arr = t_rel_s.to_numpy()
    midquote_arr = midquote.to_numpy()
    
    assert (np.diff(timedeltas_arr) >= 0).all(), "Series are not sorted by time!!!"

    ohlc = _make_time_bars_ohlc(timedeltas_arr, midquote_arr, bar_t, window)

    if ffill:
        ohlc = _ffill_ohlc(ohlc)

    return ohlc
