from dataclasses import dataclass, field
from typing import Any, Callable, Union

import pandas as pd

from .generators.base import BaseGenerator
from .transformations.base import BaseTransformation


@dataclass
class ColumnSpec:
    """Specification for a single output column."""

    name: str
    generator: BaseGenerator
    transformations: list[BaseTransformation] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class _DAGColumnSpec:
    """Internal specification for a DAGPipeline column."""

    name: str
    generator: Union[BaseGenerator, Callable]
    transformations: list[BaseTransformation] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)


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


class DAGPipeline:
    """
    A pipeline that supports explicit column dependencies, forming a DAG.

    Columns can declare that they depend on previously generated columns.
    The generator callable for a dependent column receives the already-built
    partial DataFrame so it can use those values to produce its own series.

    Example
    -------
    ::

        from datagen.pipeline import DAGPipeline
        from datagen.generators import NormalGenerator, ConstantGenerator

        # Simple: age is independent, score depends on age
        def score_from_age(df, n):
            return df["age"] * 0.5 + 10

        dag = (
            DAGPipeline()
            .add_column("age", NormalGenerator(mean=35, std=10, seed=0))
            .add_column("score", score_from_age, depends_on=["age"])
        )
        df = dag.generate(100)
    """

    def __init__(self) -> None:
        self._order: list[str] = []
        self._specs: dict[str, _DAGColumnSpec] = {}

    def add_column(
        self,
        name: str,
        generator: "Union[BaseGenerator, Callable]",
        transformations: list[BaseTransformation] | None = None,
        depends_on: list[str] | None = None,
    ) -> "DAGPipeline":
        """Register a column with optional dependencies.

        Parameters
        ----------
        name:
            Output column name.
        generator:
            Either a :class:`~datagen.generators.base.BaseGenerator` instance
            (independent column) or a callable ``(df: pd.DataFrame, n: int) ->
            pd.Series`` for columns that depend on previously generated data.
        transformations:
            Optional list of transformations to apply after generation.
        depends_on:
            List of column names that must be generated before this one.
        """
        spec = _DAGColumnSpec(
            name=name,
            generator=generator,
            transformations=transformations or [],
            depends_on=depends_on or [],
        )
        self._specs[name] = spec
        self._order.append(name)
        return self

    def remove_column(self, name: str) -> "DAGPipeline":
        """Remove a column by name and return self for chaining."""
        if name in self._specs:
            del self._specs[name]
            self._order = [n for n in self._order if n != name]
        return self

    @property
    def column_names(self) -> list[str]:
        return self._topological_order()

    def _topological_order(self) -> list[str]:
        """Return column names in dependency-resolved (topological) order."""
        visited: set[str] = set()
        result: list[str] = []

        def visit(node: str, path: set[str]) -> None:
            if node in path:
                raise ValueError(
                    f"Circular dependency detected involving column {node!r}."
                )
            if node in visited:
                return
            path = path | {node}
            for dep in self._specs[node].depends_on:
                if dep not in self._specs:
                    raise KeyError(
                        f"Column {node!r} depends on {dep!r}, which is not registered."
                    )
                visit(dep, path)
            visited.add(node)
            result.append(node)

        for name in self._order:
            visit(name, set())
        return result

    def to_dag(self) -> dict[str, list[str]]:
        """Return the dependency graph as ``{col_name: [dependency_names]}``."""
        return {name: list(spec.depends_on) for name, spec in self._specs.items()}

    def generate(self, n: int) -> pd.DataFrame:
        """Run all generators in dependency order and return a DataFrame."""
        if not self._specs:
            return pd.DataFrame()

        order = self._topological_order()
        partial: dict[str, Any] = {}

        for col_name in order:
            spec = self._specs[col_name]
            gen = spec.generator
            current_df = pd.DataFrame(partial) if partial else pd.DataFrame()

            if isinstance(gen, BaseGenerator):
                series = gen.generate(n)
            else:
                # Callable: (df, n) -> pd.Series
                series = gen(current_df, n)
                if not isinstance(series, pd.Series):
                    series = pd.Series(series)

            for transformation in spec.transformations:
                series = transformation.transform(series)
            series.name = col_name
            partial[col_name] = series

        return pd.DataFrame(partial)

    def __len__(self) -> int:
        return len(self._specs)

    def __repr__(self) -> str:
        cols = ", ".join(self._topological_order())
        return f"DAGPipeline(columns=[{cols}])"
