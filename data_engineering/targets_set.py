from abc import ABC, abstractmethod
import numpy as np
from typing import Type


class TargetsConstructor(ABC):
    @classmethod
    @abstractmethod
    def get_targets_list(cls) -> list[str]: pass

    @classmethod
    @abstractmethod
    def calculate(cls, row) -> dict: pass


class TargetSet:
    def __init__(self, target_constructors: list[Type[TargetsConstructor]]):
        self.target_constructors = target_constructors
        self.targets_list: list[str] = np.concatenate([
            tc.get_targets_list() for tc in self.target_constructors
        ]).tolist()

    def calculate(self, row):
        out = {}
        for tc in self.target_constructors:
            try:
                target_dict = tc.calculate(row)
            except Exception as e:
                print(f"Exception in {tc.__name__}:")   # debugging
                raise e
            out.update(target_dict)
        return out


class ExtremeDrawdownNoPosteriorRecovery(TargetsConstructor):
    """Based on Mazorra et al. (2022) and Srifa et al. (2025)."""

    @classmethod
    def _get_price_thresholds(cls) -> list[dict[str,float]]:
        return [
            # Mazorra et al. style price rug pull
            {"md": 0.9, "rc": 0.01},
            # Srifa et al. style "hard" rug pull
            {"md": 0.99, "rc": 0.005},
        ]

    @classmethod
    def _get_liq_thresholds(cls) -> list[dict[str,float]]:
        return [
            {"md": 0.99, "rc": 0.01},
        ]

    @classmethod
    def get_targets_list(cls) -> list[str]:
        out = [f"md_price_no_post_rc_{th['md']}_{th['rc']}" for th in cls._get_price_thresholds()]
        out += [f"md_liq_no_post_rc_{th['md']}_{th['rc']}" for th in cls._get_liq_thresholds()]
        return out

    @classmethod
    def calculate(cls, row) -> dict:
        max_price, max_tvl = row.get("max_price"), row.get("max_tvl")
        posterior_min_price, posterior_min_tvl = row.get("posterior_min_price"), row.get("posterior_min_tvl")
        last_price, last_tvl = row.get("last_price"), row.get("last_tvl")

        out = {}
        for price_thresholds in cls._get_price_thresholds():
            drop_threshold_price, recovery_threshold_price = price_thresholds["md"], price_thresholds["rc"]
            target_str = f"md_price_no_post_rc_{drop_threshold_price}_{recovery_threshold_price}"
            if last_price is None or max_price is None:
                out[target_str] = np.nan
            elif posterior_min_price is None or np.isclose(max_price, posterior_min_price) or np.isclose(max_price, 0):
                # no "posterior" inside the window => peak price is the last price => no scam
                out[target_str] = 0
            else:
                MD_condition_price = (max_price - posterior_min_price) >= (drop_threshold_price * max_price)
                RC_condition_price = (last_price - posterior_min_price) <= (recovery_threshold_price * (max_price - posterior_min_price))
                out[target_str] = int(MD_condition_price and RC_condition_price)

        for liq_thresholds in cls._get_liq_thresholds():
            drop_threshold_tvl, recovery_threshold_tvl = liq_thresholds["md"], liq_thresholds["rc"]
            target_str = f"md_liq_no_post_rc_{drop_threshold_tvl}_{recovery_threshold_tvl}"
            if last_tvl is None or max_tvl is None:
                out[target_str] = np.nan
            elif posterior_min_tvl is None or np.isclose(max_tvl, posterior_min_tvl) or np.isclose(max_tvl, 0):
                out[target_str] = 0
            else:
                MD_condition_liq = (max_tvl - posterior_min_tvl) >= (drop_threshold_tvl * max_tvl)
                RC_condition_liq = (last_tvl - posterior_min_tvl) <= (recovery_threshold_tvl * (max_tvl - posterior_min_tvl))
                out[target_str] = int(MD_condition_liq and RC_condition_liq)

        return out


class ExtremeDrawdownTVL(TargetsConstructor):
    """TVL-based approach from Yaremus et al. (2025)."""

    @classmethod
    def _get_thresholds(cls) -> list[float]: return [
        0.9,
        # 0.99
    ]

    @classmethod
    def get_targets_list(cls) -> list[str]:
        return [f"md_liq_{th}" for th in cls._get_thresholds()]

    @classmethod
    def calculate(cls, row) -> dict:
        max_tvl = row.get("max_tvl")
        posterior_min_tvl = row.get("posterior_min_tvl")

        out = {}
        for drop_threshold_tvl in cls._get_thresholds():
            target_str = f"md_liq_{drop_threshold_tvl}"
            if max_tvl is None:
                out[target_str] = np.nan
            elif posterior_min_tvl is None or np.isclose(max_tvl, posterior_min_tvl) or np.isclose(max_tvl, 0):
                out[target_str] = 0
            else:
                MD_condition_liq = (max_tvl - posterior_min_tvl) >= (drop_threshold_tvl * max_tvl)
                out[target_str] = int(MD_condition_liq)

        return out


class ProlongedInactivity(TargetsConstructor):
    "Based on Kalacheva et al. (2025) and Yaremus et al. (2025)."

    @classmethod
    def _get_thresholds(cls) -> list[float]: return [
        300,
        # 600
    ]

    @classmethod
    def get_targets_list(cls) -> list[str]:
        return [f"idle_{th}" for th in cls._get_thresholds()]

    @classmethod
    def calculate(cls, row) -> dict:
        longest_inactivity = row.get("longest_inactivity")
        return {
            f"idle_{inactivity_threshold}": np.nan if longest_inactivity is None else int(longest_inactivity >= inactivity_threshold)
            for inactivity_threshold in cls._get_thresholds()
        }


TARGETS = [
    ExtremeDrawdownNoPosteriorRecovery,
    ExtremeDrawdownTVL,
    ProlongedInactivity,
]
