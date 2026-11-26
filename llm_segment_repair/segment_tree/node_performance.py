"""Evaluate real-vs-synthetic performance for segment-tree nodes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from llm_segment_repair.metrics import evaluate_datasets
from llm_segment_repair.generation.schema import resolve_dataset_schema

from .builder import SegmentTreeNode

MetricFn = Callable[[pd.DataFrame, pd.DataFrame, Mapping[str, Any]], Mapping[str, Any]]

_SUPPORTED_EVALUATION_SDTYPES = {"categorical", "boolean", "numerical", "integer"}


@dataclass(frozen=True)
class SegmentTreeNodePerformance:
    """Performance payload for one segment-tree node."""

    node_id: str
    parent_node_id: str | None
    depth: int
    filters: dict[str, Any]
    real_row_count: int
    synthetic_row_count: int
    metrics: dict[str, Any]


def evaluate_node_performance(
    root: SegmentTreeNode,
    *,
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata: Mapping[str, Any],
    metric_fn: MetricFn | None = None,
) -> list[SegmentTreeNodePerformance]:
    """Evaluate synthetic quality independently at every tree node.

    Args:
        root: Segment tree root.
        real_df: Real dataset used as the node-level target.
        synthetic_df: Synthetic dataset to evaluate.
        metadata: Dataset metadata.
        metric_fn: Optional evaluator. Defaults to project SDMetrics evaluation.

    Returns:
        One performance payload per node in pre-order DFS.
    """

    evaluator = metric_fn or _default_metric_fn
    return [
        _evaluate_single_node(
            node,
            real_df=real_df,
            synthetic_df=synthetic_df,
            metadata=metadata,
            evaluator=evaluator,
        )
        for node in root.iter_dfs()
    ]


def _evaluate_single_node(
    node: SegmentTreeNode,
    *,
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata: Mapping[str, Any],
    evaluator: MetricFn,
) -> SegmentTreeNodePerformance:
    assert isinstance(real_df, pd.DataFrame), "real_df must be a pandas DataFrame"
    assert isinstance(synthetic_df, pd.DataFrame), "synthetic_df must be a pandas DataFrame"
    assert not real_df.empty, "real_df must not be empty"

    real_node_df = _filter_node_dataframe(real_df, node)
    synthetic_node_df = _filter_node_dataframe(synthetic_df, node)
    assert len(real_node_df) == node.real_row_count, (
        f"Node {node.id!r} expected {node.real_row_count} real rows, "
        f"got {len(real_node_df)}"
    )
    assert not real_node_df.empty, f"Node {node.id!r} has no real rows"

    if synthetic_node_df.empty:
        metrics = {
            "evaluation_status": "missing_synthetic_rows",
            "baseline_rows": len(real_node_df),
            "synthetic_rows": 0,
        }
    else:
        metrics = dict(evaluator(real_node_df, synthetic_node_df, metadata))
    metrics.setdefault("baseline_rows", len(real_node_df))
    metrics.setdefault("synthetic_rows", len(synthetic_node_df))
    return SegmentTreeNodePerformance(
        node_id=node.id,
        parent_node_id=node.parent_id,
        depth=node.depth,
        filters=node.filters,
        real_row_count=len(real_node_df),
        synthetic_row_count=len(synthetic_node_df),
        metrics=metrics,
    )


def _filter_node_dataframe(df: pd.DataFrame, node: SegmentTreeNode) -> pd.DataFrame:
    filtered = df
    for predicate in node.predicates:
        if filtered.empty:
            return filtered.reset_index(drop=True)
        filtered = filtered.loc[predicate.matches(filtered)]
    return filtered.reset_index(drop=True)


def filter_df_for_node(df: pd.DataFrame, node: SegmentTreeNode) -> pd.DataFrame:
    """Filter a DataFrame to rows matching a segment-tree node's predicates."""
    return _filter_node_dataframe(df, node)


def _default_metric_fn(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    schema_metadata = _metadata_for_evaluation(metadata, real_df, synthetic_df)
    schema = resolve_dataset_schema(
        dataset_name=str(schema_metadata.get("dataset_name", "segment_tree_node")),
        dataset_metadata=schema_metadata,
    )
    # Segment-tree scoring only reads overall_quality_score from metrics; the
    # slow privacy/utility metrics (NewRowSynthesis ~40s, DCR, LogisticDetection)
    # are skipped here to keep per-node evaluation fast during comb repair.
    return evaluate_datasets(
        real_df,
        synthetic_df,
        schema,
        include_privacy_and_utility=False,
    )


def _metadata_for_evaluation(
    metadata: Mapping[str, Any],
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> dict[str, Any]:
    columns = metadata.get("columns")
    assert isinstance(columns, Mapping), "metadata must include a columns mapping"
    shared_columns = [
        column
        for column, spec in columns.items()
        if column in real_df.columns
        and column in synthetic_df.columns
        and spec.get("sdtype") in _SUPPORTED_EVALUATION_SDTYPES
    ]
    assert shared_columns, "No supported metadata columns are present in both node frames"
    return {
        "dataset_name": metadata.get("dataset_name", "segment_tree_node"),
        "dataset_description": metadata.get("dataset_description", ""),
        "columns": {column: dict(columns[column]) for column in shared_columns},
    }
