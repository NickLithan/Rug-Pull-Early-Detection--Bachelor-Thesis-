from abc import ABC, abstractmethod
import numpy as np


# --------- to consider: ----------
# VWAP = doll_volume / volume
# Mean Effective Spread = mean(trade_sign_t * (p_t - m_t))
# or mean(trade_sign_t * (p_t - m_t) / m_t)
# Amihud Lambda, which is pretty much his illiquidity
# Amivest Liquidity
# Order Imbalance
# VNET (Engle and Lange 2001)
# ...
# ---------------------------------


class FeaturesConstructor(ABC):
    @property
    @abstractmethod
    def features_list(self) -> list[str]:
        pass

    @classmethod
    @abstractmethod
    def calculate(cls, series_dict: dict) -> dict:
        pass


# benchmark features 1
class CountingFeatures(FeaturesConstructor):
    @classmethod
    def get_features_list(cls): return [
        'n_transfers', 'n_swaps', 'n_buys',
        'n_lp_addition_like', 'n_lp_removal_like',
        'buy_percentage', 'sell_percentage', 'buy_sell_ratio',
    ]

    @classmethod
    def calculate(cls, series_dict: dict):
        is_trade = series_dict["is_trade"]
        is_lp_add = series_dict["is_lp_add"]
        is_lp_rem = series_dict["is_lp_rem"]
        trade_sign = series_dict["trade_sign"]

        n_buys = (is_trade & (trade_sign > 0)).sum().item()
        n_swaps = is_trade.sum().item()
        return {
            "n_transfers": len(is_trade),
            "n_swaps": n_swaps,
            "n_buys": n_buys,
            "n_lp_addition_like": is_lp_add.sum().item(),
            "n_lp_removal_like": is_lp_rem.sum().item(),
            "buy_percentage": n_buys / len(is_trade),
            "sell_percentage": (n_swaps - n_buys) / len(is_trade),
            "buy_sell_ratio": n_buys / (n_swaps - n_buys)
        }


# benchmark features 2
class PriceFeatures(FeaturesConstructor):
    @classmethod
    def get_features_list(cls): return [
        'init_price', 'max_price', 'min_price',
        'std_price', 'std_buy_price', 'std_sell_price',
        'return_1min', 'return_5min',
    ]

    @classmethod
    def calculate(cls, series_dict: dict):
        midquote = series_dict["midquote"]
        eff_price = series_dict["eff_price"]
        is_trade = series_dict["is_trade"]
        trade_sign = series_dict["trade_sign"]
        timedeltas = series_dict["timedeltas"]

        init_price = midquote.iloc[0].item()
        price_1min = midquote[timedeltas <= 60][-1]

        return {
            "init_price": init_price,
            "max_price": midquote.max().item(),
            "min_price": midquote.min().item(),
            "std_price": midquote.std().item(),
            "std_buy_price": eff_price[is_trade & (trade_sign > 0)].std().item(),
            "std_sell_price": eff_price[is_trade & (trade_sign < 0)].std().item(),
            "return_1min": (price_1min - init_price) / init_price,
            "return_1min": (midquote.iloc[-1].item() - init_price) / init_price
        }


# benchmark features 3
class LiquidityFeatures(FeaturesConstructor):
    @classmethod
    def get_features_list(cls): return [
        'supply', 'init_tvl_sol', 
        'max_volume', 'max_sol_volume', 'min_volume', 'min_sol_volume',
        'total_wsol_turnover', 'total_wsol_buy_turnover',
        'wsol_turnover_1min', 'wsol_buy_turnover_1min',
    ]

    @classmethod
    def calculate(cls, series_dict: dict):
        base_liquidity = series_dict["base_liquidity"]
        quote_liquidity = series_dict["quote_liquidity"]
        is_trade = series_dict["is_trade"]
        is_buy = series_dict["trade_sign"] > 0
        volume = series_dict["delta_base_vault"].abs()
        wsol_volume = series_dict["delta_quote_vault"].abs()
        is_1min = series_dict["timedeltas"] <= 60
        
        return {
            "supply": base_liquidity.iloc[0].item(),
            "init_tvl_sol": quote_liquidity.iloc[0].item(),
            "max_volume": volume[is_trade].max().item(),
            "max_wsol_volume": wsol_volume[is_trade].max().item(),
            "min_volume": volume[is_trade].min().item(),
            "min_wsol_volume": wsol_volume[is_trade].min().item(),
            "total_wsol_turnover": wsol_volume[is_trade].sum().item(),
            "total_wsol_buy_turnover": wsol_volume[is_trade & is_buy].sum().item(),
            "wsol_turnover_1min": wsol_volume[is_trade & is_1min].sum().item(),
            "wsol_buy_turnover_1min": wsol_volume[
                is_trade & is_buy & is_1min
            ].sum().item(),
        }


# benchmark features 4
class TimeFeatures(FeaturesConstructor):
    @classmethod
    def get_features_list(cls): return [
        'time_deployemnt2trade',
        'time2max_volume', 'time2max_wsol_volume',
        'first_buy_rel_ts', 'first_sell_rel_ts',
    ]

    @classmethod
    def calculate(cls, series_dict: dict):
        timedeltas = series_dict["timedeltas"]
        is_trade = series_dict["is_trade"]
        volume = series_dict["delta_base_vault"].abs()
        wsol_volume = series_dict["delta_quote_vault"].abs()
        trade_sign = series_dict["trade_sign"]
        
        volume_is_max = volume == volume.max()
        wsol_volume_is_max = wsol_volume == wsol_volume.max()

        return {
            "time_deployemnt2trade": -timedeltas.min().item(),
            "time2max_volume": timedeltas[is_trade & volume_is_max].iloc[0].item(),
            "time2max_wsol_volume": timedeltas[
                is_trade & wsol_volume_is_max
            ].iloc[0].item(),
            "first_buy_rel_ts": timedeltas[is_trade & (trade_sign > 0)].min().item(),
            "first_sell_rel_ts": timedeltas[is_trade & (trade_sign < 0)].min().item(),
        }


# benchmark features 5
class RSIFeatures(FeaturesConstructor):
    @classmethod
    def get_features_list(cls): return [
        'average_rsi', 'std_rsi',
    ]

    @classmethod
    def calculate(cls, series_dict: dict):
        
        
        return {}


# Kyle's (1985) lambda; following Lopez de Prado (2018)
class KyleLambda(FeaturesConstructor):
    @classmethod
    def get_features_list(cls): return ['kyle_lambda']

    @classmethod
    def calculate(cls, series_dict: dict, scale: float=1e12):
        is_trade = series_dict["is_trade"]
        midquote = series_dict["midquote"][is_trade].iloc[1:]
        prev_midquote = series_dict["midquote"].shift(1)[is_trade].iloc[1:]
        # bought = -(increase in vault):
        signed_volume = -series_dict["delta_base_vault"][is_trade].iloc[1:]

        # scale is used in case lambdas are too tiny
        delta_midquote_scaled = (midquote - prev_midquote) * scale
        covar = np.cov(delta_midquote_scaled, signed_volume, ddof=1)[0,1]

        var_volume = np.var(signed_volume, ddof=1)
        if var_volume < 1e-30:  # effectively dividing by 0
            return {"kyle_lambda": np.nan}

        return {"kyle_lambda": covar / var_volume}


# Harris (1990) version of Roll's (1984) Spread Estimator
class RollSpread(FeaturesConstructor):
    @classmethod
    def get_features_list(cls): return ['roll_spread', 'roll_percentage_spread']

    @classmethod
    def calculate(cls, series_dict: dict, scale: float=1e10):
        is_trade = series_dict["is_trade"]
        midquote = series_dict["midquote"][is_trade].iloc[1:]
        prev_midquote = series_dict["midquote"].shift(1)[is_trade].iloc[1:]

        delta_midquote_scaled = midquote - prev_midquote
        returns = delta_midquote_scaled / prev_midquote
        
        scov = np.cov(delta_midquote_scaled[1:], delta_midquote_scaled.shift(1)[1:], ddof=1)[0,1]
        r_spread = 2 * np.sqrt(-scov) if scov < 0 else -2 * np.sqrt(scov)

        # percentage spread estimate; normalized via scaling up by 100
        scov_percent = np.cov(returns[1:], returns.shift(1)[1:], ddof=1)[0,1]
        r_percentage_spread = 200 * np.sqrt(-scov_percent) if scov_percent < 0 else -200 * np.sqrt(scov_percent)

        return {
            "roll_spread": r_spread.item() * scale,
            "roll_percentage_spread": r_percentage_spread.item(),
        }


# TAQ-based Amihud's (2002) Illiquidity Measure
class AmihudIlliquidity(FeaturesConstructor):
    @classmethod
    def get_features_list(cls): return ['amihud_illiquidity']

    @classmethod
    def calculate(cls, series_dict: dict, scale: float=1e6) -> dict:
        is_trade = series_dict["is_trade"]
        midquote = series_dict["midquote"][is_trade].iloc[1:]
        prev_midquote = series_dict["midquote"].shift(1)[is_trade].iloc[1:]
        sol_volume = np.abs(series_dict["delta_quote_vault"][is_trade].iloc[1:]) # "dollar" volume in SOL

        if np.any(sol_volume < 1e-30) or np.any(prev_midquote < 1e-30):
            return {"amihud_illiquidity": np.nan}

        absolute_return = np.abs((midquote - prev_midquote) / prev_midquote)
        return {"amihud_illiquidity": np.mean(absolute_return * scale / sol_volume).item()}


# VPIN based on Easley et al. (2012)
class VPIN(FeaturesConstructor):
    @classmethod
    def get_features_list(cls): return ['vpin']

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

        if n_buckets < 5: # reasonable lower limit
            return {"vpin": np.nan}

        vpin = mean_absolute_order_imbalance / vol_per_bar
        return {"vpin": vpin}


# Features that can be calculated from deltas
class DeltasFeatureSet:
    def __init__(self):
        self.feature_constructors = [
            # benchmark via deltas
            CountingFeatures,
            PriceFeatures,
            LiquidityFeatures,
            TimeFeatures,
            RSIFeatures,
            # market microstructure
            KyleLambda,
            RollSpread,
            AmihudIlliquidity,
            VPIN,
        ]
        self.features_list = np.concatenate([
            fc.get_features_list() for fc in self.feature_constructors
        ]).tolist()

    def calculate(self, series_dict):
        out = {}
        for fc in self.feature_constructors:
            feature_dict = fc.calculate(series_dict)
            out.update(feature_dict)
        return out

