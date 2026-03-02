![2023-08-23_20-53-25](https://github.com/jaun-van-heerden/datagen/assets/39254276/bc2bced9-4f1d-4c96-aca7-511ac3279557)


# DataGen Platform by Jaun van Heerden

DataGen is a **modular synthetic data generation platform** built on a composable framework of generators, transformations, and pipelines. It can produce any kind of synthetic data — numeric, text, boolean, datetime — with a Streamlit UI for interactive exploration.

## Architecture

```
src/
├── datagen/                  # Core Python package
│   ├── generators/           # Pluggable data generators
│   │   ├── numeric.py        # RandomWalk, Sinusoidal, Uniform, Normal, Constant
│   │   ├── boolean.py        # RandomBoolean, WeightedBoolean
│   │   ├── text.py           # Name, Email, UUID, Category, LoremIpsum
│   │   └── datetime_gen.py   # DateRange, RandomDate
│   ├── transformations/      # Pluggable transformations
│   │   └── common.py         # Scale, Normalize, Round, Clip, AddNoise, NullInjection, Upper/Lowercase
│   ├── pipeline.py           # Pipeline — chains generators + transformations into a DataFrame
│   └── registry.py           # Central registry for generators and transformations
├── app.py                    # Streamlit platform UI
├── datagen-basic.py          # Original basic Streamlit app (preserved)
└── datagen.py                # Original advanced Streamlit prototype (preserved)
tests/
├── test_generators.py
├── test_transformations.py
└── test_pipeline.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the Platform UI

```bash
streamlit run src/app.py
```

## Use as a Python Library

```python
from datagen import Pipeline
from datagen.generators import (
    NameGenerator, NormalGenerator, CategoryGenerator,
    DateRangeGenerator, UUIDGenerator,
)
from datagen.transformations import RoundTransformation, NullInjectionTransformation

pipeline = (
    Pipeline()
    .add_column("id",         UUIDGenerator(seed=42))
    .add_column("name",       NameGenerator(seed=42))
    .add_column("age",        NormalGenerator(mean=35, std=10, seed=42),
                              transformations=[RoundTransformation(0)])
    .add_column("department", CategoryGenerator(["Engineering", "Sales", "HR"], seed=42))
    .add_column("joined",     DateRangeGenerator(unit="days"))
    .add_column("score",      NormalGenerator(mean=75, std=15, seed=42),
                              transformations=[NullInjectionTransformation(null_rate=0.05)])
)

df = pipeline.generate(n=1000)
df.to_csv("synthetic_employees.csv", index=False)
```

## Run Tests

```bash
pytest tests/ -v
```

## Features

| Feature | Detail |
|---|---|
| **Generators** | Numeric (random walk, sinusoidal, uniform, normal, constant), Boolean (random, weighted), Text (name, email, UUID, category, lorem ipsum), DateTime (date range, random date) |
| **Transformations** | Scale, Normalize, Round, Clip, Add Noise, Null Injection, Uppercase, Lowercase |
| **Pipeline** | Chain any number of generators and transformations; generate a `pd.DataFrame` in one call |
| **Registry** | Central `GENERATOR_REGISTRY` and `TRANSFORMATION_REGISTRY` for dynamic discovery |
| **Streamlit UI** | Interactive pipeline builder with Preview, Chart, Schema, and SQL DDL tabs |
| **Extensible** | Subclass `BaseGenerator` or `BaseTransformation` and add to the registry |

## Extending the Platform

### Add a custom generator

```python
from datagen.generators.base import BaseGenerator
import pandas as pd

class PhoneNumberGenerator(BaseGenerator):
    name = "Phone Number"
    description = "Generates random US-style phone numbers."
    category = "text"

    def generate(self, n: int, **kwargs) -> pd.Series:
        import random
        rng = random.Random(getattr(self, "seed", None))
        return pd.Series(
            [f"+1-{rng.randint(200,999)}-{rng.randint(100,999)}-{rng.randint(1000,9999)}"
             for _ in range(n)],
            dtype="string",
        )

# Register it
from datagen.registry import GENERATOR_REGISTRY
GENERATOR_REGISTRY[PhoneNumberGenerator.name] = PhoneNumberGenerator
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Developed by Jaun van Heerden. Contributions, feedback, and issue reports are welcome.
