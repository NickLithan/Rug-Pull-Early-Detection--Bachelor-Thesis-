from abc import ABC, abstractmethod
import numpy as np
from typing import Type


class FeaturesConstructor(ABC):
    @classmethod
    @abstractmethod
    def get_features_list(cls) -> list[str]: pass

    @classmethod
    @abstractmethod
    def calculate(cls, series_dict: dict) -> dict: pass


class DeltasFeatureSet:
    def __init__(self, feature_constructors: list[Type[FeaturesConstructor]]):
        self.feature_constructors = feature_constructors
        self.features_list: list[str] = np.concatenate([
            fc.get_features_list() for fc in self.feature_constructors
        ]).tolist()

    def calculate(self, series_dict: dict):
        out = {}
        for fc in self.feature_constructors:
            try:
                feature_dict = fc.calculate(series_dict)
            except Exception as e:
                print(f"Exception in {fc.__name__}:")   # debugging
                raise e
            out.update(feature_dict)
        return out


############################################################
#                     BENCHMARK FEATURES
############################################################

class CountingFeatures(FeaturesConstructor):
    """Numbers of event occurrences and their ratios."""

    @classmethod
    def get_features_list(cls) -> list[str]: return [
        'n_transfers', 'n_swaps', 'n_buys',
        'n_lp_addition_like', 'n_lp_removal_like',
        'buy_percentage', 'sell_percentage', 'buy_sell_ratio',
    ]

    @classmethod
    def calculate(cls, series_dict: dict) -> dict:
        is_trade = series_dict["is_trade"]
        is_lp_add = series_dict["is_lp_add"]
        is_lp_rem = series_dict["is_lp_rem"]
        trade_sign = series_dict["trade_sign"]

        n_buys = (is_trade & (trade_sign > 0)).sum()
        n_swaps = is_trade.sum()
        return {
            "n_transfers": len(is_trade),
            "n_swaps": n_swaps,
            "n_buys": n_buys,
            "n_lp_addition_like": is_lp_add.sum(),
            "n_lp_removal_like": is_lp_rem.sum(),
            "buy_percentage": n_buys / len(is_trade),
            "sell_percentage": (n_swaps - n_buys) / len(is_trade),
            "buy_sell_ratio": n_buys / (n_swaps - n_buys) if n_buys < n_swaps else np.nan
        }


class PriceFeatures(FeaturesConstructor):
    """Price snapshots and dynamics."""

    @classmethod
    def get_features_list(cls) -> list[str]: return [
        'init_price', 'max_price', 'min_price',
        'std_price', 'std_buy_price', 'std_sell_price',
        'return_1min', 'return_5min',
    ]

    @classmethod
    def calculate(cls, series_dict: dict) -> dict:
        midquote = series_dict["midquote"]
        eff_price = series_dict["eff_price"]
        is_trade = series_dict["is_trade"]
        trade_sign = series_dict["trade_sign"]
        t_rel_s = series_dict["t_rel_s"]

        init_price = midquote.iloc[0]
        price_1min = midquote[t_rel_s <= 60].iloc[-1]

        return {
            "init_price": init_price,
            "max_price": midquote.max(),
            "min_price": midquote.min(),
            "std_price": midquote.std(),
            "std_buy_price": eff_price[is_trade & (trade_sign > 0)].std(),
            "std_sell_price": eff_price[is_trade & (trade_sign < 0)].std(),
            "return_1min": (price_1min - init_price) / init_price,
            "return_5min": (midquote.iloc[-1] - init_price) / init_price
        }


class LiquidityFeatures(FeaturesConstructor):
    """Benchmark features from volume, supply, and WSOL TVL."""

    @classmethod
    def get_features_list(cls) -> list[str]: return [
        'supply', 'init_tvl_sol', 
        'max_volume', 'max_wsol_volume', 'min_volume', 'min_wsol_volume',
        'total_wsol_turnover', 'total_wsol_buy_turnover',
        'wsol_turnover_1min', 'wsol_buy_turnover_1min',
    ]

    @classmethod
    def calculate(cls, series_dict: dict) -> dict:
        base_liquidity = series_dict["base_liquidity"]
        quote_liquidity = series_dict["quote_liquidity"]
        is_trade = series_dict["is_trade"]
        is_buy = series_dict["trade_sign"] > 0
        volume = series_dict["delta_base_vault"].abs()
        wsol_volume = series_dict["delta_quote_vault"].abs()
        is_1min = series_dict["t_rel_s"] <= 60

        return {
            "supply": base_liquidity.iloc[0],
            "init_tvl_sol": quote_liquidity.iloc[0],
            "max_volume": volume[is_trade].max(),
            "max_wsol_volume": wsol_volume[is_trade].max(),
            "min_volume": volume[is_trade].min(),
            "min_wsol_volume": wsol_volume[is_trade].min(),
            "total_wsol_turnover": wsol_volume[is_trade].sum(),
            "total_wsol_buy_turnover": wsol_volume[is_trade & is_buy].sum(),
            "wsol_turnover_1min": wsol_volume[is_trade & is_1min].sum(),
            "wsol_buy_turnover_1min": wsol_volume[
                is_trade & is_buy & is_1min
            ].sum(),
        }


class TimeFeatures(FeaturesConstructor):
    """Benchmark features from time relative to first trade."""

    @classmethod
    def get_features_list(cls) -> list[str]: return [
        'time_deployemnt2trade',
        'time2max_volume', 'time2max_wsol_volume',
        'first_buy_rel_ts', 'first_sell_rel_ts',
    ]

    @classmethod
    def calculate(cls, series_dict: dict) -> dict:
        t_rel_s = series_dict["t_rel_s"]
        is_trade = series_dict["is_trade"]
        volume = np.where(is_trade, series_dict["delta_base_vault"].abs(), np.nan)
        wsol_volume = np.where(is_trade, series_dict["delta_quote_vault"].abs(), np.nan)
        trade_sign = series_dict["trade_sign"]

        volume_is_max = np.isclose(volume, volume[is_trade].max())
        wsol_volume_is_max = np.isclose(wsol_volume, wsol_volume[is_trade].max())

        return {
            "time_deployemnt2trade": -t_rel_s.min(),
            "time2max_volume": t_rel_s[is_trade & volume_is_max].iloc[0],
            "time2max_wsol_volume": t_rel_s[is_trade & wsol_volume_is_max].iloc[0],
            "first_buy_rel_ts": t_rel_s[is_trade & (trade_sign > 0)].min(),
            "first_sell_rel_ts": t_rel_s[is_trade & (trade_sign < 0)].min(),
        }


class RSIFeatures(FeaturesConstructor):
    """Benchmark features from Relative Strength Index, as in Wilder (1978)."""  

    @classmethod
    def _get_values_N(cls) -> list[int]: return [6, 12]

    @classmethod
    def get_features_list(cls) -> list[str]:
        out = []
        for N in cls._get_values_N():
            out.append(f"avg_rsi_{N}")
            out.append(f"std_rsi_{N}")
        return out

    @classmethod
    def calculate(cls, series_dict: dict) -> dict:
        midquote = series_dict["midquote"]
        t_rel_s = series_dict["t_rel_s"]

        # 1. making time bars:
        time_bar_delta = 10
        assert 300 % time_bar_delta == 0, "Time bars must evenly split the 5-minute window."

        # binary search
        close_times = np.arange(0, 300 + time_bar_delta, time_bar_delta)
        order = np.argsort(t_rel_s.values)
        timedeltas_arr = t_rel_s.values[order]
        midquote_arr = midquote.values[order]
        close_idx = np.searchsorted(timedeltas_arr, close_times, side='right') - 1
        assert np.all(close_idx >= 0), "Binary search of time bar timestamps failed."
        close_arr = midquote_arr[close_idx]
        
        # undefined values
        if not np.any(close_arr > 0):
            out = {}
            for N in cls._get_values_N():
                out[f"avg_rsi_{N}"] = np.nan
                out[f"std_rsi_{N}"] = np.nan
            return out

        # 2. Wilder's RSI via SMMA:
        # rescaling in case of small midquotes (cancels out later on)
        close_arr_scaled = close_arr / close_arr[close_arr > 0][0]
        diff = np.diff(close_arr_scaled)
        U = np.maximum(diff, 0)
        D = np.maximum(-diff, 0)

        out = {}
        for N in cls._get_values_N():   # smoothing parameter
            assert N <= len(U), "Smoothing parameter is too large."
            prev_mult, next_mult = (N-1) / N, 1 / N
            avg_gain, avg_loss = [np.mean(U[:N])], [np.mean(D[:N])]
            for i in range(N, len(U)):
                avg_gain.append(prev_mult * avg_gain[-1] + next_mult * U[i])
                avg_loss.append(prev_mult * avg_loss[-1] + next_mult * D[i])

            avg_gain, avg_loss = np.array(avg_gain), np.array(avg_loss)
            RS = np.divide(avg_gain, avg_loss,
                           out=np.full_like(avg_gain, np.nan),
                           where=~np.isclose(avg_loss, 0))  # prevent division by 0
            RSI = 100 - 100 / (1 + RS)

            # convention: loss is 0, then 100
            RSI = np.where(np.isclose(avg_loss, 0), 100.0, RSI)
            # whenever both are 0, we have to put nan
            RSI = np.where(np.isclose(avg_gain, 0) & np.isclose(avg_loss, 0), np.nan, RSI)

            out[f"avg_rsi_{N}"] = np.nanmean(RSI)
            out[f"std_rsi_{N}"] = np.nanstd(RSI)
        return out


############################################################
#              MARKET MICROSTRUCTUTRE FEATURES
############################################################

class KyleLambda(FeaturesConstructor):
    """Kyle's (1985) lambda, following Lopez de Prado (2018)."""

    @classmethod
    def get_features_list(cls) -> list[str]: return ['kyle_lambda']

    @classmethod
    def calculate(cls, series_dict: dict, scale: float=1e12) -> dict:
        is_trade = series_dict["is_trade"]
        midquote = series_dict["midquote"][is_trade].iloc[1:]
        prev_midquote = series_dict["midquote"].shift(1)[is_trade].iloc[1:]
        # bought = -(increase in vault):
        signed_volume = -series_dict["delta_base_vault"][is_trade].iloc[1:]

        # scale is used in case lambdas are too tiny
        delta_midquote_scaled = (midquote - prev_midquote) * scale
        covar = np.cov(delta_midquote_scaled, signed_volume, ddof=1)[0,1]

        var_volume = np.var(signed_volume, ddof=1)
        if np.isclose(var_volume, 0):
            return {"kyle_lambda": np.nan}

        return {"kyle_lambda": covar / var_volume}


class RollSpread(FeaturesConstructor):
    """Harris (1990) version of Roll's (1984) Spread Estimator."""

    @classmethod
    def get_features_list(cls) -> list[str]: return ['roll_spread', 'roll_percentage_spread']

    @classmethod
    def calculate(cls, series_dict: dict, scale: float=1e10) -> dict:
        is_trade = series_dict["is_trade"]
        midquote = series_dict["midquote"][is_trade].iloc[1:]
        prev_midquote = series_dict["midquote"].shift(1)[is_trade].iloc[1:]

        delta_midquote_scaled = midquote - prev_midquote
        scov = np.cov(delta_midquote_scaled[1:], delta_midquote_scaled.shift(1)[1:], ddof=1)[0,1]
        r_spread = 2 * np.sqrt(-scov) if scov < 0 else -2 * np.sqrt(scov)

        if np.any(np.isclose(prev_midquote, 0)):
            return { "roll_spread": r_spread * scale, "roll_percentage_spread": np.nan}

        # percentage spread estimate; normalized via scaling up by 100
        returns = delta_midquote_scaled / prev_midquote
        scov_percent = np.cov(returns[1:], returns.shift(1)[1:], ddof=1)[0,1]
        r_percentage_spread = 200 * np.sqrt(-scov_percent) if scov_percent < 0 else -200 * np.sqrt(scov_percent)

        return {
            "roll_spread": r_spread * scale,
            "roll_percentage_spread": r_percentage_spread,
        }


class AmihudIlliquidity(FeaturesConstructor):
    """TAQ-based Amihud's (2002) Illiquidity Measure."""

    @classmethod
    def get_features_list(cls) -> list[str]: return ['amihud_illiquidity']

    @classmethod
    def calculate(cls, series_dict: dict, scale: float=1e6) -> dict:
        is_trade = series_dict["is_trade"]
        midquote = series_dict["midquote"][is_trade].iloc[1:]
        prev_midquote = series_dict["midquote"].shift(1)[is_trade].iloc[1:]
        sol_volume = np.abs(series_dict["delta_quote_vault"][is_trade].iloc[1:])    # "dollar" volume in SOL

        if np.any(sol_volume < 1e-30) or np.any(prev_midquote < 1e-30):
            return {"amihud_illiquidity": np.nan}

        absolute_return = np.abs((midquote - prev_midquote) / prev_midquote)
        return {"amihud_illiquidity": np.mean(absolute_return * scale / sol_volume)}


class VPIN(FeaturesConstructor):
    """Variation of VPIN: Easley et al. (2012)."""

    @classmethod
    def get_features_list(cls) -> list[str]: return ['vpin']

    @classmethod
    def calculate(cls, series_dict: dict) -> dict:
        is_trade = series_dict["is_trade"]
        signed_volume_arr = -series_dict["delta_base_vault"][is_trade].values

        # we set the volume such that we can expect 10 buckets for representative trades
        TARGETED_N_BARS = 10
        targeted_trade_per_bar = len(signed_volume_arr) / TARGETED_N_BARS
        vol_per_bar = targeted_trade_per_bar * np.median(np.abs(signed_volume_arr))

        # order imbalance accumulation within volume bars
        # (trades are split between bins in case of overflow)
        v_left = vol_per_bar
        n_buckets = 0
        mean_absolute_order_imbalance = 0.0
        current_order_imbalance = 0.0

        for signed_v in signed_volume_arr:
            v, sign = abs(signed_v), np.sign(signed_v)
            
            if v < v_left:
                current_order_imbalance += signed_v
                v_left -= v
            else:
                # finalize bar
                current_order_imbalance += v_left * sign
                v -= v_left
                full_buckets, residual = int(v // vol_per_bar), v % vol_per_bar

                # update via running mean
                mean_absolute_order_imbalance *= n_buckets / (n_buckets + 1 + full_buckets)
                n_buckets += 1 + full_buckets
                mean_absolute_order_imbalance += abs(current_order_imbalance) / n_buckets 
                mean_absolute_order_imbalance += full_buckets / n_buckets * vol_per_bar

                # left over
                current_order_imbalance = sign * residual
                v_left = vol_per_bar - residual

        if n_buckets < 5:   # reasonable lower limit
            return {"vpin": np.nan}

        vpin = mean_absolute_order_imbalance / vol_per_bar
        return {"vpin": vpin}


############################################################
#                         REGISTRY
############################################################


# benchmark features that come from vault deltas
BENCHMARK_FEATURES = [
    CountingFeatures,
    PriceFeatures,
    LiquidityFeatures,
    TimeFeatures,
    RSIFeatures,
]


# market microstructure metrics
MICROSTRUCTURE_FEATURES = [
    KyleLambda,
    RollSpread,
    AmihudIlliquidity,
    VPIN,
]


# static (categorical) features
STATIC_FEATURES = [
    "pool_type",
    "has_pumpdotfun_history",
    "token_decimals",
    # "token_program",
]
