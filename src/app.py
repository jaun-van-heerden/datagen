"""
DataGen Platform — Streamlit UI
A modular synthetic data generation platform.
"""

import sys
from pathlib import Path

# Ensure the datagen package is importable when running from /src
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

from datagen import Pipeline
from datagen.registry import GENERATOR_REGISTRY, TRANSFORMATION_REGISTRY
from datagen.generators import (
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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DataGen Platform",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "columns" not in st.session_state:
    st.session_state.columns = []  # list of dict configs


# ---------------------------------------------------------------------------
# Helper: build a generator instance from UI config
# ---------------------------------------------------------------------------

def build_generator(cfg: dict):
    gen_name = cfg["generator"]
    seed = cfg.get("seed")

    if gen_name == RandomWalkGenerator.name:
        return RandomWalkGenerator(cfg["lower"], cfg["upper"], seed)
    if gen_name == SinusoidalGenerator.name:
        return SinusoidalGenerator(cfg["lower"], cfg["upper"], cfg.get("frequency", 1.0), cfg.get("phase", 0.0), seed)
    if gen_name == UniformRandomGenerator.name:
        return UniformRandomGenerator(cfg["lower"], cfg["upper"], seed)
    if gen_name == NormalGenerator.name:
        return NormalGenerator(cfg.get("mean", 0.0), cfg.get("std", 1.0), seed)
    if gen_name == ConstantGenerator.name:
        return ConstantGenerator(cfg.get("value", 0.0))
    if gen_name == RandomBooleanGenerator.name:
        return RandomBooleanGenerator(seed)
    if gen_name == WeightedBooleanGenerator.name:
        return WeightedBooleanGenerator(cfg.get("true_prob", 0.5), seed)
    if gen_name == CategoryGenerator.name:
        cats = [c.strip() for c in cfg.get("categories", "A,B,C").split(",") if c.strip()]
        return CategoryGenerator(cats, seed)
    if gen_name == NameGenerator.name:
        return NameGenerator(seed)
    if gen_name == EmailGenerator.name:
        return EmailGenerator(seed)
    if gen_name == UUIDGenerator.name:
        return UUIDGenerator(seed)
    if gen_name == LoremIpsumGenerator.name:
        return LoremIpsumGenerator(cfg.get("words", 8), seed)
    if gen_name == DateRangeGenerator.name:
        return DateRangeGenerator(unit=cfg.get("unit", "minutes"))
    if gen_name == RandomDateGenerator.name:
        return RandomDateGenerator(seed=seed)
    raise ValueError(f"Unknown generator: {gen_name}")


def build_transformations(cfg: dict) -> list:
    transforms = []
    for t_name in cfg.get("transformations", []):
        if t_name == ScaleTransformation.name:
            transforms.append(ScaleTransformation(cfg.get("scale_factor", 1.0)))
        elif t_name == NormalizeTransformation.name:
            transforms.append(NormalizeTransformation())
        elif t_name == RoundTransformation.name:
            transforms.append(RoundTransformation(cfg.get("round_decimals", 2)))
        elif t_name == ClipTransformation.name:
            transforms.append(ClipTransformation(cfg.get("clip_lower"), cfg.get("clip_upper")))
        elif t_name == AddNoiseTransformation.name:
            transforms.append(AddNoiseTransformation(cfg.get("noise_std", 0.1)))
        elif t_name == NullInjectionTransformation.name:
            transforms.append(NullInjectionTransformation(cfg.get("null_rate", 0.05)))
        elif t_name == UppercaseTransformation.name:
            transforms.append(UppercaseTransformation())
        elif t_name == LowercaseTransformation.name:
            transforms.append(LowercaseTransformation())
    return transforms


# ---------------------------------------------------------------------------
# Generator parameter widgets (rendered inside an expander)
# ---------------------------------------------------------------------------

NUMERIC_GENERATORS = [
    RandomWalkGenerator.name,
    SinusoidalGenerator.name,
    UniformRandomGenerator.name,
    NormalGenerator.name,
    ConstantGenerator.name,
]

BOOLEAN_GENERATORS = [RandomBooleanGenerator.name, WeightedBooleanGenerator.name]

TEXT_GENERATORS = [
    CategoryGenerator.name,
    NameGenerator.name,
    EmailGenerator.name,
    UUIDGenerator.name,
    LoremIpsumGenerator.name,
]

DATETIME_GENERATORS = [DateRangeGenerator.name, RandomDateGenerator.name]

ALL_GENERATORS = NUMERIC_GENERATORS + BOOLEAN_GENERATORS + TEXT_GENERATORS + DATETIME_GENERATORS

NUMERIC_TRANSFORMS = [
    ScaleTransformation.name,
    NormalizeTransformation.name,
    RoundTransformation.name,
    ClipTransformation.name,
    AddNoiseTransformation.name,
    NullInjectionTransformation.name,
]
TEXT_TRANSFORMS = [
    UppercaseTransformation.name,
    LowercaseTransformation.name,
    NullInjectionTransformation.name,
]


def generator_params_widget(col_idx: int, cfg: dict):
    gen = cfg.get("generator", RandomWalkGenerator.name)

    if gen in (RandomWalkGenerator.name, UniformRandomGenerator.name):
        cfg["lower"] = st.number_input("Lower Bound", value=cfg.get("lower", -10.0), key=f"lower_{col_idx}")
        cfg["upper"] = st.number_input("Upper Bound", value=cfg.get("upper", 10.0), key=f"upper_{col_idx}")

    elif gen == SinusoidalGenerator.name:
        cfg["lower"] = st.number_input("Lower Bound", value=cfg.get("lower", -1.0), key=f"lower_{col_idx}")
        cfg["upper"] = st.number_input("Upper Bound", value=cfg.get("upper", 1.0), key=f"upper_{col_idx}")
        cfg["frequency"] = st.slider("Frequency", 1.0, 20.0, float(cfg.get("frequency", 1.0)), key=f"freq_{col_idx}")
        cfg["phase"] = st.slider("Phase Offset", 0.0, 6.28, float(cfg.get("phase", 0.0)), step=0.1, key=f"phase_{col_idx}")

    elif gen == NormalGenerator.name:
        cfg["mean"] = st.number_input("Mean", value=cfg.get("mean", 0.0), key=f"mean_{col_idx}")
        cfg["std"] = st.number_input("Std Dev", value=cfg.get("std", 1.0), min_value=0.0001, key=f"std_{col_idx}")

    elif gen == ConstantGenerator.name:
        cfg["value"] = st.number_input("Constant Value", value=cfg.get("value", 0.0), key=f"val_{col_idx}")

    elif gen == WeightedBooleanGenerator.name:
        cfg["true_prob"] = st.slider("P(True)", 0.0, 1.0, float(cfg.get("true_prob", 0.5)), step=0.01, key=f"tp_{col_idx}")

    elif gen == CategoryGenerator.name:
        cfg["categories"] = st.text_input(
            "Categories (comma-separated)", value=cfg.get("categories", "A,B,C"), key=f"cats_{col_idx}"
        )

    elif gen == LoremIpsumGenerator.name:
        cfg["words"] = st.slider("Words per entry", 3, 30, int(cfg.get("words", 8)), key=f"words_{col_idx}")

    elif gen == DateRangeGenerator.name:
        cfg["unit"] = st.selectbox(
            "Time unit", ["seconds", "minutes", "hours", "days", "weeks"],
            index=["seconds", "minutes", "hours", "days", "weeks"].index(cfg.get("unit", "minutes")),
            key=f"unit_{col_idx}",
        )


def transformation_params_widget(col_idx: int, cfg: dict):
    selected = cfg.get("transformations", [])
    gen = cfg.get("generator", "")

    if gen in NUMERIC_GENERATORS:
        available = NUMERIC_TRANSFORMS
    elif gen in TEXT_GENERATORS:
        available = TEXT_TRANSFORMS
    else:
        available = [NullInjectionTransformation.name]

    cfg["transformations"] = st.multiselect(
        "Transformations", available, default=selected, key=f"transforms_{col_idx}"
    )

    if ScaleTransformation.name in cfg["transformations"]:
        cfg["scale_factor"] = st.number_input("Scale factor", value=cfg.get("scale_factor", 1.0), key=f"sf_{col_idx}")
    if RoundTransformation.name in cfg["transformations"]:
        cfg["round_decimals"] = st.number_input("Decimal places", value=cfg.get("round_decimals", 2), step=1, min_value=0, key=f"rd_{col_idx}")
    if ClipTransformation.name in cfg["transformations"]:
        cfg["clip_lower"] = st.number_input("Clip lower", value=cfg.get("clip_lower", -10.0), key=f"cl_{col_idx}")
        cfg["clip_upper"] = st.number_input("Clip upper", value=cfg.get("clip_upper", 10.0), key=f"cu_{col_idx}")
    if AddNoiseTransformation.name in cfg["transformations"]:
        cfg["noise_std"] = st.number_input("Noise std", value=cfg.get("noise_std", 0.1), min_value=0.0, key=f"ns_{col_idx}")
    if NullInjectionTransformation.name in cfg["transformations"]:
        cfg["null_rate"] = st.slider("Null rate", 0.0, 1.0, float(cfg.get("null_rate", 0.05)), step=0.01, key=f"nr_{col_idx}")


# ---------------------------------------------------------------------------
# Sidebar — pipeline builder
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🧬 DataGen Platform")
    st.caption("Build a pipeline of modular data generators and transformations.")
    st.divider()

    n_rows = st.number_input("Number of rows", min_value=1, max_value=100_000, value=100, step=10)

    st.subheader("Pipeline columns")

    add_col = st.button("➕ Add column", use_container_width=True)
    if add_col:
        st.session_state.columns.append(
            {"name": f"column_{len(st.session_state.columns) + 1}", "generator": RandomWalkGenerator.name, "seed": 42, "transformations": []}
        )

    cols_to_remove = []
    for i, cfg in enumerate(st.session_state.columns):
        with st.expander(f"📊 {cfg.get('name', f'column_{i+1}')}", expanded=False):
            cfg["name"] = st.text_input("Column name", value=cfg.get("name", f"column_{i+1}"), key=f"name_{i}")
            cfg["generator"] = st.selectbox("Generator", ALL_GENERATORS,
                                            index=ALL_GENERATORS.index(cfg.get("generator", RandomWalkGenerator.name)),
                                            key=f"gen_{i}")
            cfg["seed"] = st.number_input("Seed (optional)", value=int(cfg.get("seed", 42) or 42), step=1, key=f"seed_{i}")

            generator_params_widget(i, cfg)
            st.markdown("**Transformations**")
            transformation_params_widget(i, cfg)

            if st.button("🗑️ Remove", key=f"remove_{i}"):
                cols_to_remove.append(i)

    for idx in reversed(cols_to_remove):
        st.session_state.columns.pop(idx)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("🧬 DataGen Platform")
st.markdown(
    "A modular, extensible synthetic data generation platform. "
    "Build pipelines by adding columns in the sidebar, choose a generator and optional transformations, then preview or download your dataset."
)

if not st.session_state.columns:
    st.info("👈 Add columns in the sidebar to start building your pipeline.")
else:
    pipeline = Pipeline()
    errors = []
    for cfg in st.session_state.columns:
        try:
            gen = build_generator(cfg)
            transforms = build_transformations(cfg)
            pipeline.add_column(cfg["name"], gen, transforms)
        except Exception as exc:
            errors.append(f"Column **{cfg['name']}**: {exc}")

    if errors:
        for err in errors:
            st.error(err)
    else:
        try:
            df = pipeline.generate(int(n_rows))

            tab_preview, tab_chart, tab_schema, tab_sql = st.tabs(["📋 Preview", "📈 Chart", "🗂 Schema", "🛢 SQL"])

            with tab_preview:
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False)
                st.download_button("⬇️ Download CSV", csv, "datagen_output.csv", "text/csv", use_container_width=True)

            with tab_chart:
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                if numeric_cols:
                    st.line_chart(df[numeric_cols])
                else:
                    st.info("No numeric columns to chart.")

            with tab_schema:
                schema_rows = [
                    {"Column": col.name, "Generator": col.generator.name,
                     "Transformations": ", ".join(t.name for t in col.transformations) or "—",
                     "Dtype": str(df[col.name].dtype)}
                    for col in pipeline._columns
                ]
                st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)

            with tab_sql:
                table_name = st.text_input("Table name", "synthetic_data")
                dtype_map = {
                    "float64": "FLOAT",
                    "int64": "INT",
                    "bool": "BOOLEAN",
                    "object": "TEXT",
                    "string": "TEXT",
                    "datetime64[ns]": "TIMESTAMP",
                }
                col_defs = ",\n    ".join(
                    f"{col} {dtype_map.get(str(df[col].dtype), 'TEXT')}" for col in df.columns
                )
                ddl = f"CREATE TABLE {table_name} (\n    {col_defs}\n);"
                st.code(ddl, language="sql")

        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
