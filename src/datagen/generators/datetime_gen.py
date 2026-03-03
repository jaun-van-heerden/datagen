from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .base import BaseGenerator


class DateRangeGenerator(BaseGenerator):
    """Generates a sequence of evenly-spaced datetime values."""

    name = "Date Range"
    description = "Produces a sequence of evenly-spaced timestamps over a given range."
    category = "datetime"

    _UNIT_MAP = {
        "seconds": "seconds",
        "minutes": "minutes",
        "hours": "hours",
        "days": "days",
        "weeks": "weeks",
    }

    def __init__(
        self,
        start: datetime | None = None,
        unit: str = "minutes",
    ):
        self.start = start or datetime(2024, 1, 1)
        if unit not in self._UNIT_MAP:
            raise ValueError(f"unit must be one of {list(self._UNIT_MAP)}")
        self.unit = unit

    def generate(self, n: int, **kwargs) -> pd.Series:
        delta = timedelta(**{self._UNIT_MAP[self.unit]: 1})
        timestamps = [self.start + delta * i for i in range(n)]
        return pd.Series(timestamps, dtype="datetime64[ns]")


class RandomDateGenerator(BaseGenerator):
    """Generates random datetime values within a given range."""

    name = "Random Date"
    description = "Generates random timestamps uniformly distributed between two dates."
    category = "datetime"

    def __init__(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        seed: int | None = None,
    ):
        self.start = start or datetime(2020, 1, 1)
        self.end = end or datetime(2024, 12, 31)
        self.seed = seed

    def generate(self, n: int, **kwargs) -> pd.Series:
        rng = np.random.default_rng(self.seed)
        start_ts = self.start.timestamp()
        end_ts = self.end.timestamp()
        random_ts = rng.uniform(start_ts, end_ts, n)
        return pd.Series(
            [datetime.fromtimestamp(ts) for ts in random_ts],
            dtype="datetime64[ns]",
        )
