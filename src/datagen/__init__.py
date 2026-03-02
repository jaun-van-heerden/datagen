"""
DataGen — synthetic data generation framework.

Quick-start example::

    from datagen import Pipeline
    from datagen.generators import NameGenerator, NormalGenerator, CategoryGenerator
    from datagen.transformations import RoundTransformation

    pipeline = (
        Pipeline()
        .add_column("name", NameGenerator(seed=42))
        .add_column("age", NormalGenerator(mean=35, std=10, seed=42),
                    transformations=[RoundTransformation(0)])
        .add_column("department", CategoryGenerator(["Engineering", "Sales", "HR"], seed=42))
    )

    df = pipeline.generate(n=100)
    print(df.head())
"""

from .pipeline import Pipeline, ColumnSpec
from .registry import GENERATOR_REGISTRY, TRANSFORMATION_REGISTRY, get_generator, get_transformation

__all__ = [
    "Pipeline",
    "ColumnSpec",
    "GENERATOR_REGISTRY",
    "TRANSFORMATION_REGISTRY",
    "get_generator",
    "get_transformation",
]
