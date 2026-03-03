"""Tests for transformation modules."""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datagen.transformations import (
    AddNoiseTransformation,
    ClipTransformation,
    LowercaseTransformation,
    NormalizeTransformation,
    NullInjectionTransformation,
    RoundTransformation,
    ScaleTransformation,
    UppercaseTransformation,
)


NUMERIC_SERIES = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
TEXT_SERIES = pd.Series(["hello", "world", "foo", "bar"], dtype="string")


class TestScaleTransformation:
    def test_doubles(self):
        result = ScaleTransformation(factor=2.0).transform(NUMERIC_SERIES)
        assert result.tolist() == [2.0, 4.0, 6.0, 8.0, 10.0]

    def test_identity(self):
        result = ScaleTransformation(factor=1.0).transform(NUMERIC_SERIES)
        assert result.tolist() == NUMERIC_SERIES.tolist()


class TestNormalizeTransformation:
    def test_range(self):
        result = NormalizeTransformation().transform(NUMERIC_SERIES)
        assert abs(result.min()) < 1e-9
        assert abs(result.max() - 1.0) < 1e-9

    def test_constant_series(self):
        s = pd.Series([5.0, 5.0, 5.0])
        result = NormalizeTransformation().transform(s)
        assert (result == 0.0).all()


class TestRoundTransformation:
    def test_rounds_to_int(self):
        s = pd.Series([1.567, 2.345, 3.999])
        result = RoundTransformation(decimals=0).transform(s)
        assert result.tolist() == [2.0, 2.0, 4.0]

    def test_rounds_to_two_decimals(self):
        s = pd.Series([1.12345, 2.98765])
        result = RoundTransformation(decimals=2).transform(s)
        assert result.tolist() == [1.12, 2.99]


class TestClipTransformation:
    def test_clips_lower(self):
        result = ClipTransformation(lower=3.0).transform(NUMERIC_SERIES)
        assert result.min() >= 3.0

    def test_clips_upper(self):
        result = ClipTransformation(upper=3.0).transform(NUMERIC_SERIES)
        assert result.max() <= 3.0

    def test_clips_both(self):
        result = ClipTransformation(lower=2.0, upper=4.0).transform(NUMERIC_SERIES)
        assert result.min() >= 2.0
        assert result.max() <= 4.0


class TestAddNoiseTransformation:
    def test_length_preserved(self):
        result = AddNoiseTransformation(std=0.1, seed=0).transform(NUMERIC_SERIES)
        assert len(result) == len(NUMERIC_SERIES)

    def test_values_differ(self):
        result = AddNoiseTransformation(std=1.0, seed=99).transform(NUMERIC_SERIES)
        # Noise should change at least one value
        assert not (result == NUMERIC_SERIES).all()


class TestNullInjectionTransformation:
    def test_length_preserved(self):
        s = pd.Series(range(100), dtype=float)
        result = NullInjectionTransformation(null_rate=0.1, seed=0).transform(s)
        assert len(result) == 100

    def test_nulls_introduced(self):
        s = pd.Series(range(1000), dtype=float)
        result = NullInjectionTransformation(null_rate=0.2, seed=42).transform(s)
        assert result.isna().sum() > 0

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            NullInjectionTransformation(null_rate=1.5)

    def test_zero_rate_no_nulls(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = NullInjectionTransformation(null_rate=0.0, seed=0).transform(s)
        assert result.isna().sum() == 0


class TestUppercaseTransformation:
    def test_uppercase(self):
        result = UppercaseTransformation().transform(TEXT_SERIES)
        assert all(v == v.upper() for v in result)


class TestLowercaseTransformation:
    def test_lowercase(self):
        s = pd.Series(["HELLO", "WORLD"], dtype="string")
        result = LowercaseTransformation().transform(s)
        assert all(v == v.lower() for v in result)
