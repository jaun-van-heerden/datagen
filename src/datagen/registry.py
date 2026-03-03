"""
Central registry that maps human-readable names to generator and
transformation classes, making it easy to discover and instantiate
components dynamically (e.g. from a UI or config file).
"""

from .generators import (
    BaseGenerator,
    CategoryGenerator,
    ConstantGenerator,
    DateRangeGenerator,
    EmailGenerator,
    LoremIpsumGenerator,
    NameGenerator,
    NormalGenerator,
    RandomBooleanGenerator,
    RandomDateGenerator,
    RandomWalkGenerator,
    SinusoidalGenerator,
    UUIDGenerator,
    UniformRandomGenerator,
    WeightedBooleanGenerator,
)
from .transformations import (
    AddNoiseTransformation,
    BaseTransformation,
    ClipTransformation,
    LowercaseTransformation,
    NormalizeTransformation,
    NullInjectionTransformation,
    RoundTransformation,
    ScaleTransformation,
    UppercaseTransformation,
)

# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

GENERATOR_REGISTRY: dict[str, type[BaseGenerator]] = {
    # Numeric
    RandomWalkGenerator.name: RandomWalkGenerator,
    SinusoidalGenerator.name: SinusoidalGenerator,
    UniformRandomGenerator.name: UniformRandomGenerator,
    NormalGenerator.name: NormalGenerator,
    ConstantGenerator.name: ConstantGenerator,
    # Boolean
    RandomBooleanGenerator.name: RandomBooleanGenerator,
    WeightedBooleanGenerator.name: WeightedBooleanGenerator,
    # Text
    CategoryGenerator.name: CategoryGenerator,
    NameGenerator.name: NameGenerator,
    EmailGenerator.name: EmailGenerator,
    UUIDGenerator.name: UUIDGenerator,
    LoremIpsumGenerator.name: LoremIpsumGenerator,
    # DateTime
    DateRangeGenerator.name: DateRangeGenerator,
    RandomDateGenerator.name: RandomDateGenerator,
}

# ---------------------------------------------------------------------------
# Transformation registry
# ---------------------------------------------------------------------------

TRANSFORMATION_REGISTRY: dict[str, type[BaseTransformation]] = {
    ScaleTransformation.name: ScaleTransformation,
    NormalizeTransformation.name: NormalizeTransformation,
    RoundTransformation.name: RoundTransformation,
    ClipTransformation.name: ClipTransformation,
    AddNoiseTransformation.name: AddNoiseTransformation,
    NullInjectionTransformation.name: NullInjectionTransformation,
    UppercaseTransformation.name: UppercaseTransformation,
    LowercaseTransformation.name: LowercaseTransformation,
}


def get_generator(name: str) -> type[BaseGenerator]:
    """Return the generator class registered under *name*."""
    try:
        return GENERATOR_REGISTRY[name]
    except KeyError:
        raise KeyError(f"No generator named {name!r}. Available: {list(GENERATOR_REGISTRY)}")


def get_transformation(name: str) -> type[BaseTransformation]:
    """Return the transformation class registered under *name*."""
    try:
        return TRANSFORMATION_REGISTRY[name]
    except KeyError:
        raise KeyError(f"No transformation named {name!r}. Available: {list(TRANSFORMATION_REGISTRY)}")
