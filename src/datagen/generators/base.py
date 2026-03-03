from abc import ABC, abstractmethod
import pandas as pd


class BaseGenerator(ABC):
    """Abstract base class for all data generators."""

    name: str = ""
    description: str = ""
    category: str = "general"

    @abstractmethod
    def generate(self, n: int, **kwargs) -> pd.Series:
        """Generate n rows of synthetic data and return as a pandas Series."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
