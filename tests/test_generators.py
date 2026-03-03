"""Tests for all generator modules."""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datagen.generators import (
    ConstantGenerator,
    NormalGenerator,
    RandomWalkGenerator,
    SinusoidalGenerator,
    UniformRandomGenerator,
    RandomBooleanGenerator,
    WeightedBooleanGenerator,
    CategoryGenerator,
    NameGenerator,
    EmailGenerator,
    UUIDGenerator,
    LoremIpsumGenerator,
    DateRangeGenerator,
    RandomDateGenerator,
)


# ---------------------------------------------------------------------------
# Numeric generators
# ---------------------------------------------------------------------------

class TestRandomWalkGenerator:
    def test_length(self):
        gen = RandomWalkGenerator()
        assert len(gen.generate(100)) == 100

    def test_bounds(self):
        gen = RandomWalkGenerator(lower_bound=0.0, upper_bound=5.0, seed=0)
        s = gen.generate(500)
        assert s.min() >= 0.0
        assert s.max() <= 5.0

    def test_reproducible(self):
        gen = RandomWalkGenerator(seed=42)
        assert gen.generate(50).tolist() == gen.generate(50).tolist()

    def test_dtype(self):
        assert RandomWalkGenerator().generate(10).dtype == float


class TestSinusoidalGenerator:
    def test_length(self):
        assert len(SinusoidalGenerator().generate(100)) == 100

    def test_bounds(self):
        gen = SinusoidalGenerator(lower_bound=2.0, upper_bound=8.0)
        s = gen.generate(200)
        assert s.min() >= 2.0 - 1e-9
        assert s.max() <= 8.0 + 1e-9

    def test_dtype(self):
        assert SinusoidalGenerator().generate(10).dtype == float


class TestUniformRandomGenerator:
    def test_length(self):
        assert len(UniformRandomGenerator().generate(50)) == 50

    def test_bounds(self):
        gen = UniformRandomGenerator(lower_bound=3.0, upper_bound=7.0, seed=1)
        s = gen.generate(1000)
        assert s.min() >= 3.0
        assert s.max() <= 7.0

    def test_reproducible(self):
        gen = UniformRandomGenerator(seed=7)
        assert gen.generate(20).tolist() == gen.generate(20).tolist()


class TestNormalGenerator:
    def test_length(self):
        assert len(NormalGenerator().generate(100)) == 100

    def test_mean_approx(self):
        gen = NormalGenerator(mean=10.0, std=0.01, seed=0)
        s = gen.generate(1000)
        assert abs(s.mean() - 10.0) < 0.1

    def test_dtype(self):
        assert NormalGenerator().generate(10).dtype == float


class TestConstantGenerator:
    def test_all_same(self):
        gen = ConstantGenerator(value=3.14)
        s = gen.generate(50)
        assert (s == 3.14).all()

    def test_length(self):
        assert len(ConstantGenerator().generate(20)) == 20


# ---------------------------------------------------------------------------
# Boolean generators
# ---------------------------------------------------------------------------

class TestRandomBooleanGenerator:
    def test_length(self):
        assert len(RandomBooleanGenerator().generate(100)) == 100

    def test_dtype(self):
        assert RandomBooleanGenerator().generate(10).dtype == bool

    def test_reproducible(self):
        gen = RandomBooleanGenerator(seed=3)
        assert gen.generate(20).tolist() == gen.generate(20).tolist()


class TestWeightedBooleanGenerator:
    def test_all_true(self):
        gen = WeightedBooleanGenerator(true_probability=1.0, seed=0)
        assert gen.generate(50).all()

    def test_all_false(self):
        gen = WeightedBooleanGenerator(true_probability=0.0, seed=0)
        assert not gen.generate(50).any()

    def test_invalid_probability(self):
        with pytest.raises(ValueError):
            WeightedBooleanGenerator(true_probability=1.5)

    def test_length(self):
        assert len(WeightedBooleanGenerator().generate(100)) == 100


# ---------------------------------------------------------------------------
# Text generators
# ---------------------------------------------------------------------------

class TestCategoryGenerator:
    def test_length(self):
        assert len(CategoryGenerator().generate(100)) == 100

    def test_only_valid_categories(self):
        cats = ["X", "Y", "Z"]
        gen = CategoryGenerator(categories=cats, seed=0)
        s = gen.generate(200)
        assert set(s.unique()).issubset(set(cats))

    def test_default_categories(self):
        s = CategoryGenerator(seed=0).generate(100)
        assert set(s.unique()).issubset({"A", "B", "C"})


class TestNameGenerator:
    def test_length(self):
        assert len(NameGenerator().generate(50)) == 50

    def test_has_space(self):
        s = NameGenerator(seed=0).generate(20)
        assert all(" " in name for name in s)

    def test_reproducible(self):
        gen = NameGenerator(seed=10)
        assert gen.generate(10).tolist() == gen.generate(10).tolist()


class TestEmailGenerator:
    def test_length(self):
        assert len(EmailGenerator().generate(50)) == 50

    def test_has_at_sign(self):
        s = EmailGenerator(seed=0).generate(20)
        assert all("@" in email for email in s)


class TestUUIDGenerator:
    def test_length(self):
        assert len(UUIDGenerator().generate(20)) == 20

    def test_unique(self):
        s = UUIDGenerator().generate(100)
        assert s.nunique() == 100

    def test_format(self):
        s = UUIDGenerator(seed=0).generate(5)
        import re
        pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
        for val in s:
            assert pattern.match(val), f"Invalid UUID: {val}"


class TestLoremIpsumGenerator:
    def test_length(self):
        assert len(LoremIpsumGenerator().generate(30)) == 30

    def test_ends_with_period(self):
        s = LoremIpsumGenerator(seed=0).generate(10)
        assert all(str(val).endswith(".") for val in s)


# ---------------------------------------------------------------------------
# DateTime generators
# ---------------------------------------------------------------------------

class TestDateRangeGenerator:
    def test_length(self):
        assert len(DateRangeGenerator().generate(50)) == 50

    def test_monotonic(self):
        s = DateRangeGenerator(unit="days").generate(100)
        assert s.is_monotonic_increasing

    def test_invalid_unit(self):
        with pytest.raises(ValueError):
            DateRangeGenerator(unit="fortnight")


class TestRandomDateGenerator:
    def test_length(self):
        assert len(RandomDateGenerator().generate(50)) == 50

    def test_within_range(self):
        from datetime import datetime
        start = datetime(2022, 1, 1)
        end = datetime(2022, 12, 31)
        gen = RandomDateGenerator(start=start, end=end, seed=0)
        s = gen.generate(100)
        assert s.min() >= pd.Timestamp(start)
        assert s.max() <= pd.Timestamp(end)
