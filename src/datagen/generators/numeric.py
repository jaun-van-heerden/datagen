import numpy as np
import pandas as pd

from .base import BaseGenerator


class RandomWalkGenerator(BaseGenerator):
    """Generates a bounded random walk series."""

    name = "Random Walk"
    description = "Produces a random walk constrained within lower and upper bounds."
    category = "numeric"

    def __init__(self, lower_bound: float = -10.0, upper_bound: float = 10.0, seed: int | None = None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.seed = seed

    def generate(self, n: int, **kwargs) -> pd.Series:
        rng = np.random.default_rng(self.seed)
        start = (self.upper_bound + self.lower_bound) / 2
        step_magnitude = (self.upper_bound - self.lower_bound) / 10
        steps = rng.normal(0, step_magnitude, n)
        series = np.cumsum(steps) + start
        series = np.clip(series, self.lower_bound, self.upper_bound)
        return pd.Series(series, dtype=float)


class SinusoidalGenerator(BaseGenerator):
    """Generates a sinusoidal (wave) series scaled to given bounds."""

    name = "Sinusoidal"
    description = "Produces a sine wave scaled between lower and upper bounds."
    category = "numeric"

    def __init__(
        self,
        lower_bound: float = -1.0,
        upper_bound: float = 1.0,
        frequency: float = 1.0,
        phase_offset: float = 0.0,
        seed: int | None = None,
    ):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.frequency = frequency
        self.phase_offset = phase_offset
        self.seed = seed

    def generate(self, n: int, **kwargs) -> pd.Series:
        x = np.linspace(0, self.frequency * 2 * np.pi, n)
        y = np.sin(x + self.phase_offset)
        y_min, y_max = y.min(), y.max()
        if y_max != y_min:
            y = self.lower_bound + (y - y_min) / (y_max - y_min) * (self.upper_bound - self.lower_bound)
        return pd.Series(y, dtype=float)


class UniformRandomGenerator(BaseGenerator):
    """Generates uniformly distributed random values within given bounds."""

    name = "Uniform Random"
    description = "Draws values uniformly at random between lower and upper bounds."
    category = "numeric"

    def __init__(self, lower_bound: float = 0.0, upper_bound: float = 1.0, seed: int | None = None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.seed = seed

    def generate(self, n: int, **kwargs) -> pd.Series:
        rng = np.random.default_rng(self.seed)
        return pd.Series(rng.uniform(self.lower_bound, self.upper_bound, n), dtype=float)


class NormalGenerator(BaseGenerator):
    """Generates normally (Gaussian) distributed values."""

    name = "Normal Distribution"
    description = "Draws values from a normal distribution with configurable mean and std."
    category = "numeric"

    def __init__(self, mean: float = 0.0, std: float = 1.0, seed: int | None = None):
        self.mean = mean
        self.std = std
        self.seed = seed

    def generate(self, n: int, **kwargs) -> pd.Series:
        rng = np.random.default_rng(self.seed)
        return pd.Series(rng.normal(self.mean, self.std, n), dtype=float)


class ConstantGenerator(BaseGenerator):
    """Generates a constant numeric value for all rows."""

    name = "Constant"
    description = "Fills every row with the same constant value."
    category = "numeric"

    def __init__(self, value: float = 0.0):
        self.value = value

    def generate(self, n: int, **kwargs) -> pd.Series:
        return pd.Series([self.value] * n, dtype=float)
