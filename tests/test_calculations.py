"""Tests for CalculationModel and Calculation."""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import sympy as sp
from datagen.calculations import Calculation, CalculationModel


price, qty, discount, tax_rate = sp.symbols("price qty discount tax_rate")


class TestCalculation:
    def test_free_symbols(self):
        calc = Calculation("revenue", price * qty)
        assert calc.free_symbols == {"price", "qty"}

    def test_evaluate_scalar(self):
        calc = Calculation("revenue", price * qty)
        result = calc.evaluate({"price": 10, "qty": 5})
        assert result == 50

    def test_evaluate_series(self):
        calc = Calculation("revenue", price * qty)
        result = calc.evaluate({"price": pd.Series([10, 20]), "qty": pd.Series([5, 3])})
        expected = pd.Series([50, 60])
        pd.testing.assert_series_equal(pd.Series(result), expected)

    def test_repr(self):
        calc = Calculation("revenue", price * qty)
        assert "revenue" in repr(calc)

    def test_constant_expression(self):
        calc = Calculation("answer", sp.Integer(42))
        result = calc.evaluate({})
        assert result == 42.0


class TestCalculationModelBasics:
    def test_add_and_len(self):
        model = CalculationModel().add("revenue", price * qty)
        assert len(model) == 1

    def test_repr(self):
        model = CalculationModel().add("revenue", price * qty)
        assert "revenue" in repr(model)

    def test_calculation_names_order(self):
        model = (
            CalculationModel()
            .add("revenue", price * qty)
            .add("tax", sp.Symbol("revenue") * tax_rate)
        )
        names = model.calculation_names
        assert names.index("revenue") < names.index("tax")

    def test_add_string_expression(self):
        model = CalculationModel().add("doubled", "price * 2")
        df = pd.DataFrame({"price": [5, 10, 15]})
        result = model.evaluate(df)
        np.testing.assert_array_almost_equal(result["doubled"].values, [10, 20, 30])


class TestCalculationModelEvaluate:
    def _make_df(self):
        return pd.DataFrame({
            "price":    [10.0, 20.0, 30.0],
            "qty":      [5.0,   3.0,  2.0],
            "discount": [0.1,   0.0,  0.2],
        })

    def test_simple_evaluate(self):
        model = CalculationModel().add("revenue", price * qty)
        df = self._make_df()
        result = model.evaluate(df)
        expected = df["price"] * df["qty"]
        np.testing.assert_array_almost_equal(result["revenue"].values, expected.values)

    def test_chained_evaluate(self):
        rev = sp.Symbol("revenue")
        model = (
            CalculationModel()
            .add("revenue", price * qty)
            .add("discounted", rev * (1 - discount))
        )
        df = self._make_df()
        result = model.evaluate(df)
        assert "revenue" in result.columns
        assert "discounted" in result.columns
        expected_rev = df["price"] * df["qty"]
        expected_disc = expected_rev * (1 - df["discount"])
        np.testing.assert_array_almost_equal(result["discounted"].values, expected_disc.values)

    def test_original_columns_preserved(self):
        model = CalculationModel().add("revenue", price * qty)
        df = self._make_df()
        result = model.evaluate(df)
        assert "price" in result.columns
        assert "qty" in result.columns

    def test_empty_model(self):
        model = CalculationModel()
        df = self._make_df()
        result = model.evaluate(df)
        pd.testing.assert_frame_equal(result, df)


class TestCalculationModelDAG:
    def test_to_dag_empty(self):
        assert CalculationModel().to_dag() == {}

    def test_to_dag_no_dependencies(self):
        model = CalculationModel().add("revenue", price * qty)
        dag = model.to_dag()
        assert dag == {"revenue": []}

    def test_to_dag_with_dependencies(self):
        rev = sp.Symbol("revenue")
        model = (
            CalculationModel()
            .add("revenue", price * qty)
            .add("tax", rev * tax_rate)
        )
        dag = model.to_dag()
        assert dag["revenue"] == []
        assert "revenue" in dag["tax"]

    def test_circular_dependency_raises(self):
        a, b = sp.symbols("a b")
        model = (
            CalculationModel()
            .add("a", sp.Symbol("b") + 1)
            .add("b", sp.Symbol("a") + 1)
        )
        with pytest.raises(ValueError, match="Circular dependency"):
            model.evaluate(pd.DataFrame({"x": [1]}))


class TestCalculationModelSensitivity:
    def test_sensitivity_linear(self):
        """revenue = price * qty  => d(revenue)/d(price) = qty"""
        model = CalculationModel().add("revenue", price * qty)
        sens = model.sensitivity("revenue", "price")
        assert sens == qty

    def test_sensitivity_chained(self):
        """
        revenue = price * qty
        profit  = revenue * (1 - discount)
        d(profit)/d(price) = qty * (1 - discount)
        """
        rev = sp.Symbol("revenue")
        model = (
            CalculationModel()
            .add("revenue", price * qty)
            .add("profit", rev * (1 - discount))
        )
        sens = model.sensitivity("profit", "price")
        expected = sp.simplify(qty * (1 - discount))
        assert sp.simplify(sens - expected) == 0

    def test_sensitivity_constant(self):
        """doubled = price * 2  => d/d(qty) = 0"""
        model = CalculationModel().add("doubled", price * sp.Integer(2))
        sens = model.sensitivity("doubled", "qty")
        assert sens == 0

    def test_sensitivity_unknown_output_raises(self):
        model = CalculationModel().add("revenue", price * qty)
        with pytest.raises(KeyError, match="nonexistent"):
            model.sensitivity("nonexistent", "price")
