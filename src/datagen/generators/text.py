import uuid
import random
import string

import numpy as np
import pandas as pd

from .base import BaseGenerator


class CategoryGenerator(BaseGenerator):
    """Randomly samples from a user-supplied list of category labels."""

    name = "Category"
    description = "Randomly samples values from a predefined list of categories."
    category = "text"

    def __init__(self, categories: list[str] | None = None, seed: int | None = None):
        self.categories = categories or ["A", "B", "C"]
        self.seed = seed

    def generate(self, n: int, **kwargs) -> pd.Series:
        rng = np.random.default_rng(self.seed)
        indices = rng.integers(0, len(self.categories), n)
        return pd.Series([self.categories[i] for i in indices], dtype="string")


class NameGenerator(BaseGenerator):
    """Generates random full names from built-in first/last name lists."""

    name = "Full Name"
    description = "Generates random first and last name combinations."
    category = "text"

    _FIRST_NAMES = [
        "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
        "Iris", "Jack", "Karen", "Leo", "Mia", "Noah", "Olivia", "Paul",
        "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
        "Yara", "Zoe",
    ]
    _LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Martinez", "Wilson", "Anderson", "Taylor", "Thomas", "Moore",
        "Jackson", "White", "Harris", "Martin", "Thompson", "Young",
    ]

    def __init__(self, seed: int | None = None):
        self.seed = seed

    def generate(self, n: int, **kwargs) -> pd.Series:
        rng = random.Random(self.seed)
        names = [
            f"{rng.choice(self._FIRST_NAMES)} {rng.choice(self._LAST_NAMES)}"
            for _ in range(n)
        ]
        return pd.Series(names, dtype="string")


class EmailGenerator(BaseGenerator):
    """Generates random e-mail addresses derived from names."""

    name = "Email"
    description = "Generates synthetic e-mail addresses."
    category = "text"

    _DOMAINS = ["example.com", "test.org", "sample.net", "demo.io", "mail.co"]

    def __init__(self, seed: int | None = None):
        self.seed = seed

    def generate(self, n: int, **kwargs) -> pd.Series:
        rng = random.Random(self.seed)
        emails = [
            f"{''.join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 10)))}@{rng.choice(self._DOMAINS)}"
            for _ in range(n)
        ]
        return pd.Series(emails, dtype="string")


class UUIDGenerator(BaseGenerator):
    """Generates random UUID v4 strings."""

    name = "UUID"
    description = "Generates a unique UUID v4 for each row."
    category = "text"

    def __init__(self, seed: int | None = None):
        self.seed = seed

    def generate(self, n: int, **kwargs) -> pd.Series:
        rng = random.Random(self.seed)
        values = [str(uuid.UUID(int=rng.getrandbits(128), version=4)) for _ in range(n)]
        return pd.Series(values, dtype="string")


class LoremIpsumGenerator(BaseGenerator):
    """Generates short Lorem Ipsum placeholder sentences."""

    name = "Lorem Ipsum"
    description = "Generates random Lorem Ipsum placeholder text sentences."
    category = "text"

    _WORDS = (
        "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
        "tempor incididunt ut labore et dolore magna aliqua enim ad minim veniam "
        "quis nostrud exercitation ullamco laboris nisi aliquip ex ea commodo "
        "consequat duis aute irure reprehenderit voluptate velit esse cillum "
        "dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat non "
        "proident sunt culpa qui officia deserunt mollit anim id est laborum"
    ).split()

    def __init__(self, words_per_entry: int = 8, seed: int | None = None):
        self.words_per_entry = words_per_entry
        self.seed = seed

    def generate(self, n: int, **kwargs) -> pd.Series:
        rng = random.Random(self.seed)
        sentences = [
            " ".join(rng.choices(self._WORDS, k=self.words_per_entry)).capitalize() + "."
            for _ in range(n)
        ]
        return pd.Series(sentences, dtype="string")
