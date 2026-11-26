"""Dynamic structured output models based on dataset schema."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, create_model

from .schema import ResolvedSchema


def build_structured_output_models(
    schema: ResolvedSchema,
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Build dynamic Pydantic models for structured output.

    Returns (RowModel, RowsModel) where RowsModel has a "rows" list.
    """
    fields: dict[str, tuple[object, object]] = {}
    for col in schema.columns:
        if col in schema.boolean:
            field_type = Optional[bool]
        elif col in schema.integer:
            field_type = Optional[int]
        elif col in schema.numerical:
            field_type = Optional[float]
        else:
            field_type = Optional[str]
        fields[col] = (field_type, None)

    row_model = create_model(f"{schema.dataset_name.title()}Row", **fields)
    rows_model = create_model(
        f"{schema.dataset_name.title()}Rows",
        rows=(list[row_model], ...),
        __base__=BaseModel,
    )
    return row_model, rows_model
