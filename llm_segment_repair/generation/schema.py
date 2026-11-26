"""Dataset metadata resolution for synthetic generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedSchema:
    dataset_name: str
    dataset_description: str
    columns: list[str]
    categorical: list[str]
    numerical: list[str]
    integer: list[str]
    boolean: list[str]
    correlation: list[str]
    metadata_dict: dict[str, object]


def resolve_dataset_schema(
    *,
    dataset_name: str,
    dataset_metadata: dict[str, dict[str, object]],
) -> ResolvedSchema:
    categorical = []
    numerical = []
    columns = []
    integer = []
    boolean = []
    for col_name, col in dataset_metadata["columns"].items():
        if col["sdtype"] == "categorical":
            categorical.append(col_name)
        elif col["sdtype"] == "numerical":
            numerical.append(col_name)
        elif col["sdtype"] == "integer":
            integer.append(col_name)
        elif col["sdtype"] == "boolean":
            boolean.append(col_name)
        else:
            raise ValueError(f"Unknown sdtype: {col['sdtype']}")
        columns.append(col_name)

    return ResolvedSchema(
        dataset_name=dataset_name,
        dataset_description=dataset_metadata["dataset_description"],
        columns=columns,
        categorical=categorical,
        numerical=numerical,
        integer=integer,
        boolean=boolean,
        correlation=numerical + integer,
        metadata_dict=dataset_metadata,
    )
