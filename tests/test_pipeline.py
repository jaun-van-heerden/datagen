"""Tests for the Pipeline class."""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datagen import Pipeline
from datagen.generators import (
    ConstantGenerator,
    NameGenerator,
    NormalGenerator,
    RandomWalkGenerator,
    CategoryGenerator,
    DateRangeGenerator,
)
from datagen.transformations import RoundTransformation, NullInjectionTransformation, NormalizeTransformation


class TestPipelineBasics:
    def test_empty_pipeline_returns_empty_df(self):
        df = Pipeline().generate(100)
        assert df.empty

    def test_single_column(self):
        pipeline = Pipeline().add_column("x", ConstantGenerator(1.0))
        df = pipeline.generate(50)
        assert list(df.columns) == ["x"]
        assert len(df) == 50
        assert (df["x"] == 1.0).all()

    def test_multiple_columns(self):
        pipeline = (
            Pipeline()
            .add_column("a", ConstantGenerator(1.0))
            .add_column("b", ConstantGenerator(2.0))
        )
        df = pipeline.generate(10)
        assert set(df.columns) == {"a", "b"}
        assert len(df) == 10

    def test_column_names_property(self):
        pipeline = Pipeline().add_column("x", ConstantGenerator()).add_column("y", ConstantGenerator())
        assert pipeline.column_names == ["x", "y"]

    def test_len(self):
        pipeline = Pipeline().add_column("x", ConstantGenerator()).add_column("y", ConstantGenerator())
        assert len(pipeline) == 2

    def test_remove_column(self):
        pipeline = Pipeline().add_column("x", ConstantGenerator()).add_column("y", ConstantGenerator())
        pipeline.remove_column("x")
        assert pipeline.column_names == ["y"]

    def test_chaining_returns_pipeline(self):
        p = Pipeline()
        result = p.add_column("x", ConstantGenerator())
        assert result is p


class TestPipelineWithTransformations:
    def test_transformation_applied(self):
        pipeline = Pipeline().add_column(
            "val",
            ConstantGenerator(3.14),
            transformations=[RoundTransformation(0)],
        )
        df = pipeline.generate(10)
        assert (df["val"] == 3.0).all()

    def test_multiple_transformations_chained(self):
        pipeline = Pipeline().add_column(
            "val",
            NormalGenerator(mean=0.0, std=1.0, seed=42),
            transformations=[NormalizeTransformation(), RoundTransformation(2)],
        )
        df = pipeline.generate(100)
        assert df["val"].min() >= 0.0
        assert df["val"].max() <= 1.0

    def test_null_injection(self):
        pipeline = Pipeline().add_column(
            "val",
            NormalGenerator(seed=0),
            transformations=[NullInjectionTransformation(null_rate=0.2, seed=0)],
        )
        df = pipeline.generate(500)
        assert df["val"].isna().sum() > 0


class TestMixedTypesPipeline:
    def test_mixed_dtypes(self):
        pipeline = (
            Pipeline()
            .add_column("name", NameGenerator(seed=1))
            .add_column("score", NormalGenerator(mean=75.0, std=10.0, seed=1))
            .add_column("grade", CategoryGenerator(["A", "B", "C", "D"], seed=1))
            .add_column("timestamp", DateRangeGenerator(unit="days"))
        )
        df = pipeline.generate(20)
        assert set(df.columns) == {"name", "score", "grade", "timestamp"}
        assert len(df) == 20

    def test_repr(self):
        pipeline = Pipeline().add_column("x", ConstantGenerator()).add_column("y", ConstantGenerator())
        assert "x" in repr(pipeline)
        assert "y" in repr(pipeline)
