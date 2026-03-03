"""Tests for the FiniteStateMachineGenerator."""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datagen.generators.fsm import FiniteStateMachineGenerator


SIMPLE_TRANSITIONS = {
    "idle":    {"idle": 0.6, "running": 0.4},
    "running": {"running": 0.7, "idle": 0.2, "error": 0.1},
    "error":   {"idle": 0.8, "error": 0.2},
}
SIMPLE_STATES = ["idle", "running", "error"]


class TestFiniteStateMachineGeneratorBasics:
    def test_length(self):
        gen = FiniteStateMachineGenerator(
            states=SIMPLE_STATES,
            transitions=SIMPLE_TRANSITIONS,
            initial_state="idle",
        )
        result = gen.generate(100)
        assert len(result) == 100

    def test_values_in_states(self):
        gen = FiniteStateMachineGenerator(
            states=SIMPLE_STATES,
            transitions=SIMPLE_TRANSITIONS,
            initial_state="idle",
        )
        result = gen.generate(200)
        assert set(result.unique()).issubset(set(SIMPLE_STATES))

    def test_series_dtype_is_category(self):
        gen = FiniteStateMachineGenerator(
            states=SIMPLE_STATES,
            transitions=SIMPLE_TRANSITIONS,
            initial_state="idle",
        )
        result = gen.generate(50)
        assert str(result.dtype) == "category"

    def test_starts_at_initial_state(self):
        gen = FiniteStateMachineGenerator(
            states=SIMPLE_STATES,
            transitions=SIMPLE_TRANSITIONS,
            initial_state="error",
        )
        result = gen.generate(10)
        assert result.iloc[0] == "error"

    def test_reproducible_with_seed(self):
        gen1 = FiniteStateMachineGenerator(
            states=SIMPLE_STATES,
            transitions=SIMPLE_TRANSITIONS,
            initial_state="idle",
            seed=42,
        )
        gen2 = FiniteStateMachineGenerator(
            states=SIMPLE_STATES,
            transitions=SIMPLE_TRANSITIONS,
            initial_state="idle",
            seed=42,
        )
        assert gen1.generate(100).tolist() == gen2.generate(100).tolist()

    def test_different_seeds_differ(self):
        gen1 = FiniteStateMachineGenerator(
            states=SIMPLE_STATES,
            transitions=SIMPLE_TRANSITIONS,
            initial_state="idle",
            seed=1,
        )
        gen2 = FiniteStateMachineGenerator(
            states=SIMPLE_STATES,
            transitions=SIMPLE_TRANSITIONS,
            initial_state="idle",
            seed=99,
        )
        assert gen1.generate(100).tolist() != gen2.generate(100).tolist()


class TestFiniteStateMachineGeneratorTwoStates:
    def test_two_state_machine(self):
        gen = FiniteStateMachineGenerator(
            states=["on", "off"],
            transitions={"on": {"on": 0.5, "off": 0.5}, "off": {"on": 0.5, "off": 0.5}},
            initial_state="on",
            seed=0,
        )
        result = gen.generate(50)
        assert len(result) == 50
        assert set(result.unique()).issubset({"on", "off"})

    def test_absorbing_state(self):
        """A state with 100% self-transition — once entered, never leaves."""
        gen = FiniteStateMachineGenerator(
            states=["start", "done"],
            transitions={"start": {"done": 1.0}, "done": {"done": 1.0}},
            initial_state="start",
            seed=0,
        )
        result = gen.generate(5)
        # First element is "start", rest should be "done"
        assert result.iloc[0] == "start"
        assert (result.iloc[1:] == "done").all()


class TestFiniteStateMachineGeneratorValidation:
    def test_invalid_initial_state(self):
        with pytest.raises(ValueError, match="initial_state"):
            FiniteStateMachineGenerator(
                states=SIMPLE_STATES,
                transitions=SIMPLE_TRANSITIONS,
                initial_state="unknown",
            )

    def test_missing_transition_for_state(self):
        incomplete = {"idle": {"idle": 0.5, "running": 0.5}}
        with pytest.raises(KeyError, match="running"):
            FiniteStateMachineGenerator(
                states=["idle", "running"],
                transitions=incomplete,
                initial_state="idle",
            )

    def test_probabilities_do_not_sum_to_one(self):
        bad = {
            "idle":    {"idle": 0.5, "running": 0.3},  # sums to 0.8
            "running": {"running": 0.7, "idle": 0.3},
        }
        with pytest.raises(ValueError, match="sum to 1"):
            FiniteStateMachineGenerator(
                states=["idle", "running"],
                transitions=bad,
                initial_state="idle",
            )

    def test_empty_states(self):
        with pytest.raises(ValueError, match="non-empty"):
            FiniteStateMachineGenerator(states=[], transitions={}, initial_state="x")

    def test_transition_to_unknown_state(self):
        bad = {
            "idle": {"idle": 0.5, "ghost": 0.5},
        }
        with pytest.raises(ValueError, match="ghost"):
            FiniteStateMachineGenerator(
                states=["idle"],
                transitions=bad,
                initial_state="idle",
            )
