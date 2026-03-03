"""Tests for the DAGPipeline class."""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datagen.pipeline import DAGPipeline
from datagen.generators import ConstantGenerator, NormalGenerator, UniformRandomGenerator
from datagen.transformations import RoundTransformation


class TestDAGPipelineBasics:
    def test_empty_pipeline_returns_empty_df(self):
        df = DAGPipeline().generate(10)
        assert df.empty

    def test_single_independent_column(self):
        dag = DAGPipeline().add_column("x", ConstantGenerator(5.0))
        df = dag.generate(20)
        assert list(df.columns) == ["x"]
        assert len(df) == 20
        assert (df["x"] == 5.0).all()

    def test_multiple_independent_columns(self):
        dag = (
            DAGPipeline()
            .add_column("a", ConstantGenerator(1.0))
            .add_column("b", ConstantGenerator(2.0))
        )
        df = dag.generate(10)
        assert set(df.columns) == {"a", "b"}
        assert len(df) == 10

    def test_len(self):
        dag = DAGPipeline().add_column("x", ConstantGenerator()).add_column("y", ConstantGenerator())
        assert len(dag) == 2

    def test_remove_column(self):
        dag = DAGPipeline().add_column("x", ConstantGenerator()).add_column("y", ConstantGenerator())
        dag.remove_column("x")
        assert "x" not in dag.column_names
        assert "y" in dag.column_names

    def test_repr(self):
        dag = DAGPipeline().add_column("x", ConstantGenerator()).add_column("y", ConstantGenerator())
        r = repr(dag)
        assert "x" in r
        assert "y" in r

    def test_chaining_returns_pipeline(self):
        dag = DAGPipeline()
        result = dag.add_column("x", ConstantGenerator())
        assert result is dag


class TestDAGPipelineWithDependencies:
    def test_callable_generator_receives_df(self):
        def double_a(df, n):
            return df["a"] * 2

        dag = (
            DAGPipeline()
            .add_column("a", ConstantGenerator(3.0))
            .add_column("b", double_a, depends_on=["a"])
        )
        df = dag.generate(5)
        assert (df["b"] == 6.0).all()

    def test_dependency_order_is_respected(self):
        """Register 'b' first (depends on 'a'), then 'a'. Should still work."""
        def b_from_a(df, n):
            return df["a"] + 10

        dag = (
            DAGPipeline()
            .add_column("b", b_from_a, depends_on=["a"])
            .add_column("a", ConstantGenerator(1.0))
        )
        df = dag.generate(5)
        assert (df["b"] == 11.0).all()

    def test_chained_dependencies(self):
        """c depends on b, which depends on a."""
        def b_from_a(df, n):
            return df["a"] * 2

        def c_from_b(df, n):
            return df["b"] + 1

        dag = (
            DAGPipeline()
            .add_column("a", ConstantGenerator(4.0))
            .add_column("b", b_from_a, depends_on=["a"])
            .add_column("c", c_from_b, depends_on=["b"])
        )
        df = dag.generate(5)
        assert (df["c"] == 9.0).all()

    def test_transformation_applied_to_dependent_column(self):
        def raw_score(df, n):
            return df["a"] * 2.1

        dag = (
            DAGPipeline()
            .add_column("a", ConstantGenerator(3.0))
            .add_column("score", raw_score, depends_on=["a"],
                        transformations=[RoundTransformation(0)])
        )
        df = dag.generate(5)
        assert (df["score"] == 6.0).all()


class TestDAGPipelineDAGInspection:
    def test_to_dag_empty(self):
        assert DAGPipeline().to_dag() == {}

    def test_to_dag_no_deps(self):
        dag = DAGPipeline().add_column("x", ConstantGenerator())
        assert dag.to_dag() == {"x": []}

    def test_to_dag_with_deps(self):
        dag = (
            DAGPipeline()
            .add_column("a", ConstantGenerator())
            .add_column("b", lambda df, n: df["a"], depends_on=["a"])
        )
        graph = dag.to_dag()
        assert graph["a"] == []
        assert graph["b"] == ["a"]


class TestDAGPipelineErrors:
    def test_circular_dependency_raises(self):
        dag = (
            DAGPipeline()
            .add_column("a", lambda df, n: df["b"], depends_on=["b"])
            .add_column("b", lambda df, n: df["a"], depends_on=["a"])
        )
        with pytest.raises(ValueError, match="Circular dependency"):
            dag.generate(5)

    def test_missing_dependency_raises(self):
        dag = DAGPipeline().add_column("b", lambda df, n: df["ghost"], depends_on=["ghost"])
        with pytest.raises(KeyError):
            dag.generate(5)
