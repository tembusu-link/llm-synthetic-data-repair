import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FILTER_OPTIONS: dict[str, list[dict[str, Any]]] = {}
LLM_METADATA_COLUMNS = [
    # "latest_experiment_id",
    "latest_llm_reason",
    "latest_llm_issue_categories",
    "latest_llm_confidence",
    "latest_llm_severity",
    "latest_llm_suggested_actions",
]

DATASETS = [
    "adult_income",
    "insurance",
    "south_africa_domestic_tourism_2022_sample",
    "tourism_spending",
]


def _generate_column_label(column_name: str) -> str:
    """Generate human-readable label from column name."""
    return column_name.replace("_", " ").title()


def _load_metadata(dataset_name: str) -> dict[str, Any]:
    """Load metadata JSON file for a dataset."""
    metadata_dir = Path(__file__).parent / "data"
    metadata_file = metadata_dir / f"{dataset_name}.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file) as f:
        return json.load(f)


def _derive_columns_from_metadata(
    metadata: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Derive categorical and numerical columns from metadata based on sdtype.

    Returns:
        (categorical_columns, numerical_columns)
    """
    columns = metadata.get("columns", {})
    categorical = []
    numerical = []

    for col_name, col_spec in columns.items():
        sdtype = col_spec.get("sdtype")
        if sdtype == "categorical":
            categorical.append(col_name)
        elif sdtype == "numerical":
            numerical.append(col_name)

    return tuple(categorical), tuple(numerical)


def _generate_column_labels(columns: tuple[str, ...]) -> dict[str, str]:
    """Generate display labels for columns."""
    return {col: _generate_column_label(col) for col in columns}


@dataclass
class DatasetConfig:
    """Column and rendering configuration for a single dataset/study."""

    baseline_variant: str  # variant name of the real baseline data
    categorical_columns: tuple[str, ...]
    numerical_columns: tuple[str, ...]
    dataset_segment_columns: tuple[str, ...]
    column_labels: dict[str, str]
    dataset_description: str


def _create_dataset_config_from_metadata(dataset_name: str) -> DatasetConfig:
    """Create DatasetConfig by loading and processing metadata JSON."""
    metadata = _load_metadata(dataset_name)
    categorical_columns, numerical_columns = _derive_columns_from_metadata(metadata)

    # Generate on-demand configurations
    column_labels = _generate_column_labels(numerical_columns)

    # Use first 4 categorical columns as dataset segments if available
    dataset_segment_columns = (
        categorical_columns[:4] if len(categorical_columns) >= 4 else categorical_columns
    )

    return DatasetConfig(
        baseline_variant="real_data",
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        dataset_segment_columns=dataset_segment_columns,
        column_labels=column_labels,
        dataset_description=metadata.get("dataset_description", ""),
    )


def _load_all_datasets() -> dict[str, DatasetConfig]:
    """Auto-load all metadata files from the metadata directory."""
    datasets = {}
    metadata_dir = Path(__file__).parent / "data"

    if not metadata_dir.exists():
        return datasets

    for metadata_file in metadata_dir.glob("*.json"):
        dataset_name = metadata_file.stem
        try:
            datasets[dataset_name] = _create_dataset_config_from_metadata(dataset_name)
        except Exception as e:
            # Log warning but continue loading other datasets
            print(f"Warning: Failed to load dataset '{dataset_name}': {e}")

    return datasets


DATASETS: dict[str, DatasetConfig] = _load_all_datasets()


def get_categorical_columns(dataset_name: str) -> tuple[str, ...]:
    """Get categorical columns for a dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATASETS")
    return DATASETS[dataset_name].categorical_columns


def get_numerical_columns(dataset_name: str) -> tuple[str, ...]:
    """Get numerical columns for a dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATASETS")
    return DATASETS[dataset_name].numerical_columns


def get_column_labels(dataset_name: str) -> dict[str, str]:
    """Get column labels for a dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATASETS")
    return DATASETS[dataset_name].column_labels
