from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .generators.base import BaseGenerator
from .transformations.base import BaseTransformation


@dataclass
class ColumnSpec:
    """Specification for a single output column."""

    name: str
    generator: BaseGenerator
    transformations: list[BaseTransformation] = field(default_factory=list)


class Pipeline:
    """
    Orchestrates data generation by chaining :class:`ColumnSpec` entries.

    Example usage::

        pipeline = Pipeline()
        pipeline.add_column("age", NormalGenerator(mean=35, std=10))
        pipeline.add_column("name", NameGenerator())
        df = pipeline.generate(n=100)
    """

    def __init__(self) -> None:
        self._columns: list[ColumnSpec] = []

    def add_column(
        self,
        name: str,
        generator: BaseGenerator,
        transformations: list[BaseTransformation] | None = None,
    ) -> "Pipeline":
        """Add a column to the pipeline and return self for chaining."""
        self._columns.append(ColumnSpec(name=name, generator=generator, transformations=transformations or []))
        return self

    def remove_column(self, name: str) -> "Pipeline":
        """Remove a column by name and return self for chaining."""
        self._columns = [col for col in self._columns if col.name != name]
        return self

    @property
    def column_names(self) -> list[str]:
        return [col.name for col in self._columns]

    def generate(self, n: int) -> pd.DataFrame:
        """Run all generators (and their transformations) and return a DataFrame."""
        if not self._columns:
            return pd.DataFrame()

        data: dict[str, Any] = {}
        for spec in self._columns:
            series = spec.generator.generate(n)
            for transformation in spec.transformations:
                series = transformation.transform(series)
            series.name = spec.name
            data[spec.name] = series

        return pd.DataFrame(data)

    def __len__(self) -> int:
        return len(self._columns)

    def __repr__(self) -> str:
        cols = ", ".join(self.column_names)
        return f"Pipeline(columns=[{cols}])"
