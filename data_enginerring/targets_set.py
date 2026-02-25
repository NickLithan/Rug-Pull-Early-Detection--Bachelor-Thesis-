from abc import ABC, abstractmethod
import numpy as np


class TargetsConstructor(ABC):
    @classmethod
    @abstractmethod
    def get_features_list(cls) -> list[str]:
        pass

    @classmethod
    @abstractmethod
    def calculate(cls, series_dict: dict) -> dict:
        pass


# Mazorra
class ExtremeDrawdownNoPosteriorRecovery(TargetsConstructor):
    pass


# Srifa
class ExtremeDrawdownNoRecovery(TargetsConstructor):
    pass


# Kalacheva
class ProlongedInactivity(TargetsConstructor):
    pass


# Yaremus



