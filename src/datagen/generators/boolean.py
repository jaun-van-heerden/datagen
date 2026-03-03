import numpy as np
import pandas as pd

from .base import BaseGenerator


class RandomBooleanGenerator(BaseGenerator):
    """Generates random boolean values with equal probability."""

    name = "Random Boolean"
    description = "Generates True/False values with equal probability."
    category = "boolean"

    def __init__(self, seed: int | None = None):
        self.seed = seed

    def generate(self, n: int, **kwargs) -> pd.Series:
        rng = np.random.default_rng(self.seed)
        return pd.Series(rng.choice([True, False], n), dtype=bool)


class WeightedBooleanGenerator(BaseGenerator):
    """Generates boolean values with a configurable probability of True."""

    name = "Weighted Boolean"
    description = "Generates True/False values where the probability of True is configurable."
    category = "boolean"

    def __init__(self, true_probability: float = 0.5, seed: int | None = None):
        if not 0.0 <= true_probability <= 1.0:
            raise ValueError("true_probability must be between 0.0 and 1.0")
        self.true_probability = true_probability
        self.seed = seed

    def generate(self, n: int, **kwargs) -> pd.Series:
        rng = np.random.default_rng(self.seed)
        return pd.Series(rng.random(n) < self.true_probability, dtype=bool)
