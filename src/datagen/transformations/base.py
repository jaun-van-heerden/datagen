from abc import ABC, abstractmethod
import pandas as pd


class BaseTransformation(ABC):
    """Abstract base class for all data transformations."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def transform(self, series: pd.Series) -> pd.Series:
        """Apply transformation to a pandas Series and return the result."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
