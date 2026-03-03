"""
Finite State Machine generator.

Generates a sequence of states by randomly walking through a
user-defined state graph where each state specifies transition
probabilities to other states.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseGenerator


class FiniteStateMachineGenerator(BaseGenerator):
    """
    Generates a categorical series by simulating a Finite State Machine (FSM).

    Each state has a mapping of *{next_state: probability}*.  Transition
    probabilities for each state must sum to 1.

    Parameters
    ----------
    states:
        Sequence of valid state names (strings).
    transitions:
        Nested dict ``{state: {next_state: probability}}``.  Every state in
        *states* must have a corresponding entry.  Probabilities for each
        source state must sum to 1 (within floating-point tolerance).
    initial_state:
        The starting state.  Must be present in *states*.
    seed:
        Optional random seed for reproducibility.

    Example
    -------
    ::

        gen = FiniteStateMachineGenerator(
            states=["idle", "running", "error"],
            transitions={
                "idle":    {"idle": 0.6, "running": 0.4},
                "running": {"running": 0.7, "idle": 0.2, "error": 0.1},
                "error":   {"idle": 0.8, "error": 0.2},
            },
            initial_state="idle",
        )
        series = gen.generate(200)
    """

    name = "Finite State Machine"
    description = (
        "Simulates a Finite State Machine, producing a categorical sequence "
        "of states based on user-defined transition probabilities."
    )
    category = "categorical"

    def __init__(
        self,
        states: list[str],
        transitions: dict[str, dict[str, float]],
        initial_state: str,
        seed: int | None = None,
    ) -> None:
        if not states:
            raise ValueError("states must be a non-empty list.")
        if initial_state not in states:
            raise ValueError(f"initial_state {initial_state!r} must be one of {states}.")
        for state in states:
            if state not in transitions:
                raise KeyError(f"No transition defined for state {state!r}.")
            total = sum(transitions[state].values())
            if not abs(total - 1.0) < 1e-6:
                raise ValueError(
                    f"Transition probabilities for state {state!r} must sum to 1.0, got {total}."
                )
            for target in transitions[state]:
                if target not in states:
                    raise ValueError(
                        f"Transition target {target!r} from state {state!r} is not in states."
                    )

        self.states = states
        self.transitions = transitions
        self.initial_state = initial_state
        self.seed = seed

    def generate(self, n: int, **kwargs) -> pd.Series:
        """Run the FSM for *n* steps and return the resulting state sequence."""
        rng = np.random.default_rng(self.seed)
        result: list[str] = []
        current = self.initial_state

        for _ in range(n):
            result.append(current)
            trans = self.transitions[current]
            targets = list(trans.keys())
            probs = [trans[t] for t in targets]
            current = rng.choice(targets, p=probs)

        return pd.Series(result, dtype="category")
