"""Build a deterministic segment tree from real data and dataset metadata.

Tree construction is independent from traversal, generation, and Mongo
persistence. A later drilldown/rollup traversal can reuse the same tree for
multiple synthetic variants.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

SplitMethod = Literal["information_gain", "manual_dimension"]

_CATEGORICAL_SDTYPES = {"categorical", "boolean"}
_ENTROPY_SDTYPES = {"categorical", "boolean", "numerical", "integer"}
_MISSING_LABEL = "__MISSING__"


@dataclass(frozen=True)
class SegmentPredicate:
    """One categorical equality predicate in a node's filter conjunction."""

    column: str
    value: Any
    label: str

    def matches(self, df: pd.DataFrame) -> pd.Series:
        assert self.column in df.columns, f"Predicate column '{self.column}' missing"
        if isinstance(self.value, tuple):
            null_requested = any(pd.isna(value) for value in self.value)
            concrete_values = [value for value in self.value if not pd.isna(value)]
            mask = df[self.column].isin(concrete_values)
            if null_requested:
                mask |= df[self.column].isna()
            return mask
        if pd.isna(self.value):
            return df[self.column].isna()
        return df[self.column] == self.value

    def to_filter_value(self) -> Any:
        if isinstance(self.value, tuple):
            return list(self.value)
        return self.value


@dataclass(frozen=True)
class SegmentSplit:
    """Split metadata attached to the parent node."""

    column: str
    method: SplitMethod
    information_gain: float
    child_count: int

    @property
    def score(self) -> float:
        return self.information_gain


@dataclass
class SegmentTreeNode:
    """A node in the segment tree.

    A node's predicate is the conjunction of ``predicates``.
    """

    id: str
    depth: int
    parent_id: str | None
    predicates: tuple[SegmentPredicate, ...]
    real_row_count: int
    split: SegmentSplit | None = None
    children: list[SegmentTreeNode] = field(default_factory=list)

    @property
    def filters(self) -> dict[str, Any]:
        return {predicate.column: predicate.to_filter_value() for predicate in self.predicates}

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def iter_dfs(self) -> list[SegmentTreeNode]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.iter_dfs())
        return nodes


@dataclass(frozen=True)
class SegmentTreeConfig:
    """Configuration for deterministic tree construction.

    Automatic mode:
        Leave ``manual_dimension_order`` unset. At each node, the builder picks
        the available dimension with maximum information gain over the outcome
        distribution under that node.

    Manual mode:
        Set ``manual_dimension_order``. The builder follows the classical OLAP
        prefix lattice and splits depth ``d`` on
        ``manual_dimension_order[d]``.

    Synthetic row allocation is intentionally outside tree construction.
    """

    min_child_rows: int = 50
    manual_dimension_order: tuple[str, ...] | None = None
    sparse_bucket_label: str = "Other"


@dataclass(frozen=True)
class _CandidateSplit:
    column: str
    method: SplitMethod
    children: tuple[tuple[SegmentPredicate, pd.DataFrame], ...]
    information_gain: float


def build_segment_tree(
    real_df: pd.DataFrame,
    metadata: dict[str, Any],
    config: SegmentTreeConfig | None = None,
) -> SegmentTreeNode:
    """Construct a segment tree from real rows.

    Args:
        real_df: Real data used to define segment membership.
        metadata: Dataset metadata with a ``columns`` mapping and SDTypes.
        config: Tree-construction policy.

    Returns:
        Root ``SegmentTreeNode`` with recursively populated children.
    """

    cfg = config or SegmentTreeConfig()
    assert isinstance(real_df, pd.DataFrame), "real_df must be a pandas DataFrame"
    assert not real_df.empty, "real_df must not be empty"
    assert cfg.min_child_rows > 0, "min_child_rows must be > 0"

    split_columns, entropy_columns = _resolve_column_groups(metadata, cfg)
    for column in (*split_columns, *entropy_columns):
        assert column in real_df.columns, f"Configured column '{column}' missing from real_df"

    root = SegmentTreeNode(
        id="root",
        depth=0,
        parent_id=None,
        predicates=(),
        real_row_count=len(real_df),
    )
    _expand_node(
        root,
        real_df.reset_index(drop=True),
        split_columns,
        entropy_columns,
        cfg,
    )
    return root


def _resolve_column_groups(
    metadata: dict[str, Any],
    config: SegmentTreeConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    columns = metadata.get("columns")
    assert isinstance(columns, dict), "metadata must include a columns mapping"

    split_columns = tuple(
        column for column, spec in columns.items() if spec.get("sdtype") in _CATEGORICAL_SDTYPES
    )
    entropy_columns = tuple(
        column for column, spec in columns.items() if spec.get("sdtype") in _ENTROPY_SDTYPES
    )

    if config.manual_dimension_order is not None:
        missing = sorted(set(config.manual_dimension_order) - set(split_columns))
        assert not missing, f"manual_dimension_order contains unknown split columns: {missing}"
        split_columns = tuple(config.manual_dimension_order)

    assert split_columns, "At least one categorical or boolean split column is required"
    assert entropy_columns, "At least one entropy column is required"
    return split_columns, entropy_columns


def _expand_node(
    node: SegmentTreeNode,
    node_df: pd.DataFrame,
    split_columns: tuple[str, ...],
    entropy_columns: tuple[str, ...],
    config: SegmentTreeConfig,
) -> None:
    candidate = _choose_split(
        node=node,
        node_df=node_df,
        split_columns=split_columns,
        entropy_columns=entropy_columns,
        config=config,
    )
    if candidate is None:
        return

    node.split = SegmentSplit(
        column=candidate.column,
        method=candidate.method,
        information_gain=candidate.information_gain,
        child_count=len(candidate.children),
    )

    sibling_ids: set[str] = set()
    for predicate, child_df in candidate.children:
        child_id = _unique_child_id(node.id, predicate, sibling_ids)
        sibling_ids.add(child_id)
        child = SegmentTreeNode(
            id=child_id,
            depth=node.depth + 1,
            parent_id=node.id,
            predicates=(*node.predicates, predicate),
            real_row_count=len(child_df),
        )
        node.children.append(child)
        _expand_node(
            child,
            child_df.reset_index(drop=True),
            split_columns,
            entropy_columns,
            config,
        )


def _choose_split(
    *,
    node: SegmentTreeNode,
    node_df: pd.DataFrame,
    split_columns: tuple[str, ...],
    entropy_columns: tuple[str, ...],
    config: SegmentTreeConfig,
) -> _CandidateSplit | None:
    used_columns = {predicate.column for predicate in node.predicates}

    if config.manual_dimension_order is not None:
        if node.depth >= len(config.manual_dimension_order):
            return None
        column = config.manual_dimension_order[node.depth]
        if column in used_columns:
            return None
        return _build_dimension_split(
            node_df=node_df,
            column=column,
            entropy_columns=entropy_columns,
            min_child_rows=config.min_child_rows,
            method="manual_dimension",
            sparse_bucket_label=config.sparse_bucket_label,
        )

    candidates: list[_CandidateSplit] = []
    for column in split_columns:
        if column in used_columns:
            continue
        candidate = _build_dimension_split(
            node_df=node_df,
            column=column,
            entropy_columns=entropy_columns,
            min_child_rows=config.min_child_rows,
            method="information_gain",
            sparse_bucket_label=config.sparse_bucket_label,
        )
        if candidate is not None and candidate.information_gain > 0:
            candidates.append(candidate)

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.information_gain, item.column))


def _build_dimension_split(
    *,
    node_df: pd.DataFrame,
    column: str,
    entropy_columns: tuple[str, ...],
    min_child_rows: int,
    method: SplitMethod,
    sparse_bucket_label: str,
) -> _CandidateSplit | None:
    if column not in node_df.columns:
        return None

    child_specs: list[tuple[SegmentPredicate, pd.DataFrame]] = []
    values = node_df[column].drop_duplicates().tolist()
    values = sorted(values, key=lambda value: (pd.isna(value), str(value)))
    sparse_values: list[Any] = []
    sparse_frames: list[pd.DataFrame] = []
    for value in values:
        mask = node_df[column].isna() if pd.isna(value) else node_df[column] == value
        child_df = node_df.loc[mask].copy()
        if len(child_df) < min_child_rows:
            sparse_values.append(value)
            sparse_frames.append(child_df)
            continue
        child_specs.append(
            (
                SegmentPredicate(column=column, value=value, label=_value_label(value)),
                child_df,
            )
        )

    if sparse_frames:
        sparse_df = pd.concat(sparse_frames).sort_index()
        if len(sparse_df) < min_child_rows:
            return None
        child_specs.append(
            (
                SegmentPredicate(
                    column=column,
                    value=tuple(sparse_values),
                    label=sparse_bucket_label,
                ),
                sparse_df,
            )
        )

    if len(child_specs) < 2:
        return None
    if sum(len(child_df) for _, child_df in child_specs) != len(node_df):
        return None

    target_columns = tuple(item for item in entropy_columns if item != column)
    if not target_columns:
        return None

    information_gain = _information_gain(
        parent_df=node_df,
        child_frames=[child_df for _, child_df in child_specs],
        target_columns=target_columns,
    )
    return _CandidateSplit(
        column=column,
        method=method,
        children=tuple(child_specs),
        information_gain=information_gain,
    )


def _information_gain(
    *,
    parent_df: pd.DataFrame,
    child_frames: list[pd.DataFrame],
    target_columns: tuple[str, ...],
) -> float:
    parent_labels = _joint_target_labels(parent_df, target_columns)
    parent_entropy = _entropy(parent_labels)
    weighted_child_entropy = 0.0

    for child_df in child_frames:
        child_labels = parent_labels.loc[child_df.index]
        weighted_child_entropy += (len(child_df) / len(parent_df)) * _entropy(child_labels)

    gain = parent_entropy - weighted_child_entropy
    return max(0.0, float(gain))


def _joint_target_labels(df: pd.DataFrame, target_columns: tuple[str, ...]) -> pd.Series:
    assert target_columns, "target_columns must not be empty"
    encoded_columns: list[pd.Series] = []
    for column in target_columns:
        assert column in df.columns, f"Target column '{column}' missing"
        encoded_columns.append(_encode_outcome_column(df[column]))

    encoded = pd.concat(encoded_columns, axis=1)
    return encoded.astype(str).agg("|".join, axis=1)


def _encode_outcome_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and numeric.nunique(dropna=True) > 10:
        bin_count = max(2, min(int(round(math.sqrt(len(series)))), numeric.nunique(dropna=True)))
        try:
            bucketed = pd.qcut(numeric, q=bin_count, duplicates="drop")
            return bucketed.astype(str).where(bucketed.notna(), _MISSING_LABEL)
        except ValueError:
            pass
    return series.astype("object").where(series.notna(), _MISSING_LABEL).astype(str)


def _entropy(labels: pd.Series) -> float:
    counts = labels.value_counts(dropna=False)
    total = int(counts.sum())
    assert total > 0, "Cannot compute entropy for empty labels"
    entropy = 0.0
    for count in counts:
        probability = float(count) / float(total)
        entropy -= probability * math.log2(probability)
    return entropy


def _value_label(value: Any) -> str:
    if pd.isna(value):
        return _MISSING_LABEL
    return str(value)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.=-]+", "_", str(value)).strip("_") or "value"


def _unique_child_id(
    parent_id: str,
    predicate: SegmentPredicate,
    sibling_ids: set[str],
) -> str:
    base = f"{parent_id}/{_slugify(predicate.column)}={_slugify(predicate.label)}"
    if base not in sibling_ids:
        return base

    value_signature = _slugify(jsonish_value_label(predicate.to_filter_value()))
    candidate = f"{base}__{value_signature}"
    if candidate not in sibling_ids:
        return candidate

    suffix = 2
    while f"{candidate}_{suffix}" in sibling_ids:
        suffix += 1
    return f"{candidate}_{suffix}"


def jsonish_value_label(value: Any) -> str:
    if isinstance(value, list):
        return "_".join(_value_label(item) for item in value)
    return _value_label(value)
