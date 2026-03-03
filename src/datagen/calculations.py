"""
Model calculations with symbolic expressions and dependency resolution.

This module provides :class:`Calculation` and :class:`CalculationModel` which
allow users to define symbolic formulas (using *sympy*) and compute derived
columns over a ``pd.DataFrame``.  Dependencies between calculations are
resolved automatically via a topological sort so that each formula always
receives already-computed values.

Additionally :meth:`CalculationModel.sensitivity` exposes first-order symbolic
partial-derivative sensitivity analysis, and
:meth:`CalculationModel.to_dag` returns the dependency graph for inspection.

Example
-------
::

    from sympy import symbols
    from datagen.calculations import CalculationModel

    price, qty, discount = symbols("price qty discount")

    model = (
        CalculationModel()
        .add("revenue",  price * qty)
        .add("discounted_revenue", symbols("revenue") * (1 - discount))
        .add("tax",      symbols("discounted_revenue") * 0.15)
        .add("profit",   symbols("discounted_revenue") - symbols("tax"))
    )

    import pandas as pd
    df = pd.DataFrame({"price": [10, 20, 30], "qty": [5, 3, 2], "discount": [0.1, 0.0, 0.2]})
    result = model.evaluate(df)

    # Sensitivity of 'profit' with respect to 'price'
    sens = model.sensitivity("profit", "price")
"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    import sympy as sp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sympy is required for the calculations module. "
        "Install it with: pip install sympy"
    ) from exc


class Calculation:
    """A named symbolic calculation.

    Parameters
    ----------
    name:
        The output column name produced by this calculation.
    expression:
        A sympy expression.  Free symbols in the expression that are *not*
        the names of other :class:`Calculation` objects are treated as
        raw input-column references.
    """

    def __init__(self, name: str, expression: sp.Expr) -> None:
        self.name = name
        self.expression = expression

    @property
    def free_symbols(self) -> set[str]:
        """Return the string names of all free symbols in the expression."""
        return {str(s) for s in self.expression.free_symbols}

    def evaluate(self, context: dict[str, Any]) -> Any:
        """Numerically evaluate the expression given *context* (name → value).

        Values in *context* can be scalars or numpy arrays / pandas Series.
        Returns a scalar or Series depending on the input types.
        """
        # Build substitution dict with sympy symbols as keys
        subs = {sp.Symbol(k): v for k, v in context.items() if k in self.free_symbols}
        # Use lambdify for vectorised evaluation when arrays are present
        symbols_list = list(subs.keys())
        values_list = [subs[s] for s in symbols_list]
        if symbols_list:
            func = sp.lambdify(symbols_list, self.expression, modules="numpy")
            return func(*values_list)
        # Fully numeric expression (no free symbols left)
        return float(self.expression)

    def __repr__(self) -> str:
        return f"Calculation(name={self.name!r}, expression={self.expression})"


class CalculationModel:
    """
    A collection of :class:`Calculation` objects organised as a DAG.

    Calculations are executed in dependency order (topological sort).
    Intermediate results are added as new columns to the output DataFrame so
    that downstream calculations can reference them.

    Parameters
    ----------
    None — use :meth:`add` to register calculations.
    """

    def __init__(self) -> None:
        self._calcs: dict[str, Calculation] = {}

    # ------------------------------------------------------------------
    # Building the model
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        expression: sp.Expr | str,
    ) -> "CalculationModel":
        """Register a calculation and return *self* for chaining.

        Parameters
        ----------
        name:
            Output column / variable name.
        expression:
            A sympy expression or a string that will be parsed by
            ``sympy.sympify``.
        """
        if isinstance(expression, str):
            expression = sp.sympify(expression)
        self._calcs[name] = Calculation(name=name, expression=expression)
        return self

    # ------------------------------------------------------------------
    # DAG helpers
    # ------------------------------------------------------------------

    def to_dag(self) -> dict[str, list[str]]:
        """Return the dependency graph as ``{calc_name: [dependency_names]}``.

        Only dependencies that are *also* calculations (not raw input columns)
        are included as graph edges.
        """
        graph: dict[str, list[str]] = {}
        for name, calc in self._calcs.items():
            deps = [s for s in calc.free_symbols if s in self._calcs]
            graph[name] = deps
        return graph

    def _topological_order(self) -> list[str]:
        """Return calculation names in topological (dependency-first) order."""
        graph = self.to_dag()
        visited: set[str] = set()
        order: list[str] = []

        def visit(node: str, path: set[str]) -> None:
            if node in path:
                raise ValueError(
                    f"Circular dependency detected involving calculation {node!r}."
                )
            if node in visited:
                return
            path = path | {node}
            for dep in graph.get(node, []):
                visit(dep, path)
            visited.add(node)
            order.append(node)

        for name in graph:
            visit(name, set())
        return order

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate all calculations over *df* and return a new DataFrame.

        Input columns from *df* are available to all calculations.  Results of
        intermediate calculations are also made available to downstream ones.
        The returned DataFrame contains both the original columns and all
        computed columns.
        """
        result = df.copy()
        order = self._topological_order()
        for name in order:
            calc = self._calcs[name]
            context = {col: result[col] for col in result.columns if col in calc.free_symbols}
            result[name] = calc.evaluate(context)
        return result

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity(self, output: str, with_respect_to: str) -> sp.Expr:
        """Return the symbolic first-order partial derivative of *output*
        with respect to *with_respect_to*.

        The derivative is computed on the fully-substituted symbolic
        expression (all intermediate calculation symbols are recursively
        substituted so the result is expressed purely in terms of leaf inputs).

        Parameters
        ----------
        output:
            Name of the calculation whose sensitivity you want.
        with_respect_to:
            Name of the variable (input column or calculation) to differentiate
            against.

        Returns
        -------
        sympy.Expr
            The symbolic partial derivative, simplified.
        """
        if output not in self._calcs:
            raise KeyError(f"No calculation named {output!r}.")

        expr = self._expanded_expr(output)
        var = sp.Symbol(with_respect_to)
        return sp.simplify(sp.diff(expr, var))

    def _expanded_expr(self, name: str) -> sp.Expr:
        """Recursively substitute all intermediate calculations into *name*'s
        expression, returning it fully in terms of leaf (input) symbols.
        """
        calc = self._calcs[name]
        expr = calc.expression
        for sym_name in calc.free_symbols:
            if sym_name in self._calcs:
                sub_expr = self._expanded_expr(sym_name)
                expr = expr.subs(sp.Symbol(sym_name), sub_expr)
        return expr

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def calculation_names(self) -> list[str]:
        """Return calculation names in topological order."""
        return self._topological_order()

    def __len__(self) -> int:
        return len(self._calcs)

    def __repr__(self) -> str:
        names = list(self._calcs)
        return f"CalculationModel(calculations={names})"
