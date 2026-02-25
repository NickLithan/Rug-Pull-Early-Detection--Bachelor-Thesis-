import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from tqdm.auto import tqdm
from typing import Type, Optional
from feature_set import DeltasFeatureSet, FeaturesConstructor, STATIC_FEATURES


def decode_token_id(path: str|Path, start_ts: str, end_ts: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    pool_info = pd.read_csv('data/raydium_pools_enriched_q4_2024.csv',
                               parse_dates=['first_trade_time'])
    pool_info_splice = pool_info[
        (pool_info['first_trade_time'] >= start_ts) &
        (pool_info['first_trade_time'] < end_ts)
    ].copy()
    
    pool_info_splice = pool_info_splice.sort_values("token_mint").reset_index(drop=True)
    pool_info_splice["token_id"] = pool_info_splice.index + 1

    joined = table.set_index("token_id").join(pool_info_splice.set_index("token_id"))
    return joined


def recombine_tables(init_tables_with_start_and_end_ts: list[tuple[str, str, str]],
                     start_ts: str, end_ts: str):
    """Appends sequential tables (with no skips of tokens/timestamps in between)."""

    decoded_tables = []
    for source_path, source_start_ts, source_end_ts in init_tables_with_start_and_end_ts:
        table = decode_token_id(source_path, source_start_ts, source_end_ts)
        decoded_tables.append(table[['token_mint', 'packed_events']].reset_index(drop=True).copy())
    total_decoded_table = pd.concat(decoded_tables)
    assert not total_decoded_table["token_mint"].duplicated().any()

    pool_info = pd.read_csv('data/raydium_pools_enriched_q4_2024.csv',
                               parse_dates=['first_trade_time'])[['token_mint', 'first_trade_time']]
    pool_info_splice = pool_info[
        (pool_info['first_trade_time'] >= start_ts) &
        (pool_info['first_trade_time'] < end_ts)
    ].copy()

    pool_info_splice = pool_info_splice.sort_values("token_mint").reset_index(drop=True)
    pool_info_splice["token_id"] = pool_info_splice.index + 1

    joined = total_decoded_table.set_index("token_mint").join(pool_info_splice.set_index("token_mint"))
    return joined.reset_index(drop=True)[['token_id', 'packed_events']]


def decode_packed_events(packed_events: str, decimals: int) -> tuple[np.ndarray, ...]:
    rows = [r.split(',') for r in packed_events.split(';') if r]
    event_seq = np.fromiter((int(r[0]) for r in rows), dtype=np.int64)
    t_rel_s = np.fromiter((int(r[1]) for r in rows), dtype=np.int64)

    scale_base, scale_quote = 10 ** decimals, 10 ** 9

    delta_base_vault = np.array([int(r[2]) / scale_base for r in rows], dtype=np.float64)
    delta_quote_vault = np.array([int(r[3]) / scale_quote for r in rows], dtype=np.float64)
    return event_seq, t_rel_s, delta_base_vault, delta_quote_vault


def process_events(event_seq: np.ndarray, t_rel_s: np.ndarray,
                   delta_base_vault: np.ndarray, delta_quote_vault: np.ndarray,
                   horizon_s: int = 5 * 60):
    # make sure that the first trade ocurred when we think it did
    t_rel_s_of_first_trade = np.min(t_rel_s[(delta_base_vault * delta_quote_vault) < 0])
    t_rel_s_recalculated = t_rel_s - t_rel_s_of_first_trade

    launch_td, horizon_td = pd.to_timedelta(0, unit='s'), pd.to_timedelta(horizon_s, unit='s')
    timedeltas = pd.to_timedelta(pd.Series(t_rel_s_recalculated, index=event_seq), unit='s')
    time_mask = (timedeltas <= horizon_td)
    timedeltas = timedeltas[time_mask].sort_index()

    base_delta_series = pd.Series(delta_base_vault, index=event_seq)[time_mask].sort_index()
    quote_delta_series = pd.Series(delta_quote_vault, index=event_seq)[time_mask].sort_index()
    assert base_delta_series.notna().all() and quote_delta_series.notna().all(), "Missing values in vault deltas."

    is_trade = (base_delta_series * quote_delta_series) < 0
    assert sum(is_trade) >= 30, "Too few trades, need at least 30."

    is_post_launch = (timedeltas >= launch_td)
    is_other = (base_delta_series == 0) | (quote_delta_series == 0)
    is_lp_addition_like = (base_delta_series > 0) & (quote_delta_series > 0)
    is_lp_removal_like = (base_delta_series < 0) & (quote_delta_series < 0)

    base_liquidity_series = base_delta_series.cumsum()
    quote_liquidity_series = quote_delta_series.cumsum()

    midquote_series = quote_liquidity_series / base_liquidity_series
    effective_price = (-quote_delta_series / base_delta_series).where(is_trade)
    trade_sign = (2 * (base_delta_series < 0) - 1).where(is_trade)

    return {
        "timedeltas": timedeltas.copy(),
        "t_rel_s": timedeltas.dt.total_seconds().astype(int).copy(),
        "delta_base_vault": base_delta_series.copy(),
        "delta_quote_vault": quote_delta_series.copy(),
        "base_liquidity": base_liquidity_series.copy(),
        "quote_liquidity": quote_liquidity_series.copy(),
        "is_post_launch": is_post_launch.copy(),
        "is_trade": is_trade.copy(),
        "is_lp_add": is_lp_addition_like.copy(),
        "is_lp_rem": is_lp_removal_like.copy(),
        "is_other": is_other.copy(),
        "midquote": midquote_series.copy(),
        "eff_price": effective_price.copy(),
        "trade_sign": trade_sign.copy(),
    }


def make_features_table(feature_constructors: list[Type[FeaturesConstructor]],
                        out_csv: str, progress_desc: Optional[str] = None):
    """Calculates features from encoded event tables."""

    directory = Path("data/vault_deltas_5min")
    output_path = Path(f"data/features/{out_csv}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_set = DeltasFeatureSet(feature_constructors)
    all_rows = []
    skipped = 0

    day_files = sorted(
        f for month_dir in sorted(directory.iterdir())
        if month_dir.is_dir()
        for f in sorted(month_dir.iterdir())
        if f.suffix == ".csv"
    )

    desc_str = "Days" if not progress_desc else f"[{progress_desc}] Days"
    for day_file in tqdm(day_files, desc=desc_str):
        date_str = day_file.stem
        start_ts = f"{date_str} 00:00:00"
        end_ts = (pd.Timestamp(date_str) + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

        decoded = decode_token_id(str(day_file), start_ts, end_ts)

        for _, row in decoded.iterrows():
            packed = row.get("packed_events")
            if pd.isna(packed) or not str(packed).strip():
                skipped += 1
                continue

            token_mint = row["token_mint"]
            decimals = int(row["token_decimals"])

            try:
                event_seq, t_rel_s, db, dq = decode_packed_events(str(packed), decimals)
                series = process_events(event_seq, t_rel_s, db, dq)
                features = feature_set.calculate(series)
            except Exception as e:
                print(f"  WARN {token_mint}: {e}")
                skipped += 1
                continue

            features["token_mint"] = token_mint
            for static_feature in STATIC_FEATURES:
                features[static_feature] = row.get(static_feature)
            all_rows.append(features)

    result = pd.DataFrame(all_rows)
    meta_cols = ["token_mint"] + STATIC_FEATURES
    feat_cols = [c for c in result.columns if c not in meta_cols]
    result = result[meta_cols + feat_cols]
    result.to_csv(output_path, index=False)

    print(f"Done: {len(result)} tokens with features, {skipped} skipped -> {output_path}")
