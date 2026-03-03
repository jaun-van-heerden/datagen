import numpy as np
import pandas as pd

from .base import BaseTransformation


class ScaleTransformation(BaseTransformation):
    """Multiplies all values in the series by a constant factor."""

    name = "Scale"
    description = "Multiply every value by a constant factor."

    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def transform(self, series: pd.Series) -> pd.Series:
        return series * self.factor


class NormalizeTransformation(BaseTransformation):
    """Scales values to the [0, 1] range (min-max normalization)."""

    name = "Normalize"
    description = "Scale values to the [0, 1] range using min-max normalization."

    def transform(self, series: pd.Series) -> pd.Series:
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            return pd.Series([0.0] * len(series), index=series.index, dtype=float)
        return (series - s_min) / (s_max - s_min)


class RoundTransformation(BaseTransformation):
    """Rounds numeric values to a given number of decimal places."""

    name = "Round"
    description = "Round values to a specified number of decimal places."

    def __init__(self, decimals: int = 2):
        self.decimals = decimals

    def transform(self, series: pd.Series) -> pd.Series:
        return series.round(self.decimals)


class ClipTransformation(BaseTransformation):
    """Clips values to stay within [lower, upper] bounds."""

    name = "Clip"
    description = "Clip values to stay within lower and upper bounds."

    def __init__(self, lower: float | None = None, upper: float | None = None):
        self.lower = lower
        self.upper = upper

    def transform(self, series: pd.Series) -> pd.Series:
        return series.clip(lower=self.lower, upper=self.upper)


class AddNoiseTransformation(BaseTransformation):
    """Adds Gaussian noise to a numeric series."""

    name = "Add Noise"
    description = "Inject Gaussian noise with a configurable standard deviation."

    def __init__(self, std: float = 0.1, seed: int | None = None):
        self.std = std
        self.seed = seed

    def transform(self, series: pd.Series) -> pd.Series:
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0, self.std, len(series))
        return series + noise


class NullInjectionTransformation(BaseTransformation):
    """Randomly replaces a fraction of values with NaN/None."""

    name = "Null Injection"
    description = "Replace a random fraction of values with null (NaN/None)."

    def __init__(self, null_rate: float = 0.05, seed: int | None = None):
        if not 0.0 <= null_rate <= 1.0:
            raise ValueError("null_rate must be between 0.0 and 1.0")
        self.null_rate = null_rate
        self.seed = seed

    def transform(self, series: pd.Series) -> pd.Series:
        rng = np.random.default_rng(self.seed)
        mask = rng.random(len(series)) < self.null_rate
        result = series.copy()
        result[mask] = None
        return result


class UppercaseTransformation(BaseTransformation):
    """Converts all string values to uppercase."""

    name = "Uppercase"
    description = "Convert all string values to uppercase."

    def transform(self, series: pd.Series) -> pd.Series:
        return series.str.upper()


class LowercaseTransformation(BaseTransformation):
    """Converts all string values to lowercase."""

    name = "Lowercase"
    description = "Convert all string values to lowercase."

    def transform(self, series: pd.Series) -> pd.Series:
        return series.str.lower()
