"""Comprehensive evaluation metrics for synthetic data quality assessment.

Provides unified evaluation combining:
- Custom distribution metrics (KS, EMD, TVD)
- Correlation and SVD structural analysis
- SDMetrics library integration (privacy, utility, validity)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

# SDMetrics imports
from sdmetrics.column_pairs.statistical import (
    ContingencySimilarity,
    CorrelationSimilarity,
)
from sdmetrics.errors import ConstantInputError
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.single_table._properties import column_pair_trends
from sdmetrics.single_column import TVComplement, KSComplement
from sdmetrics.single_column.statistical import (
    BoundaryAdherence,
    CategoryAdherence,
    CategoryCoverage,
    RangeCoverage,
    StatisticSimilarity,
)
from sdmetrics.single_table.detection.sklearn import LogisticDetection
from sdmetrics.single_table.privacy.disclosure_protection import DisclosureProtection

# Project imports
from llm_segment_repair.config import get_categorical_columns, get_numerical_columns
from llm_segment_repair.generation.schema import ResolvedSchema
from llm_segment_repair.metadata import load_metadata_dict

logger = logging.getLogger(__name__)


@contextmanager
def _timer(label: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[TIMING] {label}: {elapsed:.3f}s")


def _timed(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"[TIMING] {func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper


def _convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for MongoDB serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    return obj


def _get_numeric_columns(dataset_name: str) -> list[str]:
    """Get numerical columns for a dataset."""
    return list(get_numerical_columns(dataset_name))


def _get_categorical_columns(dataset_name: str) -> list[str]:
    """Get categorical columns for a dataset."""
    return list(get_categorical_columns(dataset_name))


def _get_all_columns(dataset_name: str) -> list[str]:
    """Get all columns for a dataset."""
    return _get_numeric_columns(dataset_name) + _get_categorical_columns(dataset_name)


def kolmogorov_smirnov(real: pd.Series, synthetic: pd.Series) -> float:
    real_vals = pd.to_numeric(real, errors="coerce").dropna().to_numpy()
    syn_vals = pd.to_numeric(synthetic, errors="coerce").dropna().to_numpy()
    if real_vals.size == 0 or syn_vals.size == 0:
        return float("nan")
    real_sorted = np.sort(real_vals)
    syn_sorted = np.sort(syn_vals)
    combined = np.concatenate([real_sorted, syn_sorted])
    real_cdf = np.searchsorted(real_sorted, combined, side="right") / real_sorted.size
    syn_cdf = np.searchsorted(syn_sorted, combined, side="right") / syn_sorted.size
    return float(np.max(np.abs(real_cdf - syn_cdf)))


def earth_movers_distance(real: pd.Series, synthetic: pd.Series) -> float:
    """Compute the first Wasserstein distance (1D Earth mover) between two samples."""

    real_vals = pd.to_numeric(real, errors="coerce").dropna().to_numpy()
    syn_vals = pd.to_numeric(synthetic, errors="coerce").dropna().to_numpy()
    if real_vals.size == 0 or syn_vals.size == 0:
        return float("nan")
    assert len(real_vals) > 0, f"{real_vals=}"
    assert len(syn_vals) > 0, f"{syn_vals=}"
    real_sorted = np.sort(real_vals)
    syn_sorted = np.sort(syn_vals)
    combined = np.concatenate([real_sorted, syn_sorted])
    combined.sort()

    i = j = 0
    cum_real = cum_syn = 0.0
    inv_real = 1.0 / real_sorted.size
    inv_syn = 1.0 / syn_sorted.size
    prev = combined[0]
    emd = 0.0

    for value in combined:
        while i < real_sorted.size and real_sorted[i] <= value:
            cum_real += inv_real
            i += 1
        while j < syn_sorted.size and syn_sorted[j] <= value:
            cum_syn += inv_syn
            j += 1
        emd += abs(cum_real - cum_syn) * (value - prev)
        prev = value

    return float(emd)


def kl_divergence(
    real: pd.Series, synthetic: pd.Series, bins: int = 50, epsilon: float = 1e-6
) -> float:
    real_vals = pd.to_numeric(real, errors="coerce").dropna().to_numpy()
    syn_vals = pd.to_numeric(synthetic, errors="coerce").dropna().to_numpy()
    if real_vals.size == 0 or syn_vals.size == 0:
        return float("nan")

    vmin = min(real_vals.min(), syn_vals.min())
    vmax = max(real_vals.max(), syn_vals.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return float("nan")

    real_hist, edges = np.histogram(real_vals, bins=bins, range=(vmin, vmax), density=False)
    syn_hist, _ = np.histogram(syn_vals, bins=bins, range=(vmin, vmax), density=False)

    real_prob = real_hist.astype(np.float64) + epsilon
    syn_prob = syn_hist.astype(np.float64) + epsilon

    real_prob /= real_prob.sum()
    syn_prob /= syn_prob.sum()

    return float(np.sum(real_prob * np.log(real_prob / syn_prob)))


def summarize_numerical_columns(
    real_df: pd.DataFrame, syn_df: pd.DataFrame, columns: list[str]
) -> list[dict[str, float]]:
    """Summarize numerical column distributions with full English metric names."""
    numerical_summary: list[dict[str, float]] = []
    for column in columns:
        if column not in real_df.columns or column not in syn_df.columns:
            continue

        real_col = pd.to_numeric(real_df[column], errors="coerce").dropna()
        syn_col = pd.to_numeric(syn_df[column], errors="coerce").dropna()
        if real_col.empty or syn_col.empty:
            continue

        entry: dict[str, float] = {
            "column": column,
            "real_count": float(len(real_col)),
            "synthetic_count": float(len(syn_col)),
            "kolmogorov_smirnov_statistic": kolmogorov_smirnov(real_col, syn_col),
            "earth_movers_distance": earth_movers_distance(real_col, syn_col),
            "kullback_leibler_divergence": kl_divergence(real_col, syn_col),
            "ks_complement": KSComplement.compute_breakdown(real_col, syn_col),
        }
        entry["mean_similarity"] = StatisticSimilarity.compute_breakdown(
            real_col, syn_col, statistic="mean"
        )
        entry["standard_deviation_similarity"] = StatisticSimilarity.compute_breakdown(
            real_col, syn_col, statistic="std"
        )
        entry["boundary_adherence"] = BoundaryAdherence.compute_breakdown(
            real_col, syn_col
        )
        entry["range_coverage"] = RangeCoverage.compute_breakdown(real_col, syn_col)
        numerical_summary.append(entry)

    # Correlation similarity for numerical pairs
    # Coerce numerical columns for correlation analysis
    real_numeric = real_df[columns].apply(pd.to_numeric, errors="coerce")
    syn_numeric = syn_df[columns].apply(pd.to_numeric, errors="coerce")
    num_cols_present = [
        c for c in columns if c in real_numeric.columns and c in syn_numeric.columns
    ]
    correlation_pair = {}
    for c1, c2 in combinations(num_cols_present, 2):
        pair_key = f"{c1}__{c2}"
        if not _has_min_pairwise_correlation_samples(real_numeric, syn_numeric, c1, c2):
            logger.warning(
                "evaluation_insufficient_pairwise_correlation_samples pair=%s real_rows=%s synthetic_rows=%s",
                pair_key,
                len(real_numeric[[c1, c2]].dropna()),
                len(syn_numeric[[c1, c2]].dropna()),
            )
            continue
        real_pair = real_numeric[[c1, c2]].dropna()
        syn_pair = syn_numeric[[c1, c2]].dropna()
        if (real_pair.nunique() <= 1).any() or (syn_pair.nunique() <= 1).any():
            logger.warning("evaluation_constant_column_skipped pair=%s", pair_key)
            continue
        try:
            correlation_pair[pair_key] = CorrelationSimilarity.compute_breakdown(
                real_numeric[[c1, c2]], syn_numeric[[c1, c2]]
            )
            correlation_pair[pair_key]["key"] = pair_key
            correlation_pair[pair_key]["c1"] = c1
            correlation_pair[pair_key]["c2"] = c2
        except (ConstantInputError, Exception) as e:
            logger.warning("evaluation_correlation_error pair=%s", pair_key, exc_info=True)
    correlation_summary = correlation_gap(real_numeric, syn_numeric, columns)
    correlation_summary["details"] = list(correlation_pair.values())
    return numerical_summary, correlation_summary


CATEGORICAL_NULL_SENTINEL = "None"


def _normalize_categorical_value(value: object, sentinel: str) -> str:
    """Coerce a single cell to the canonical string form used for categorical eval."""
    if value is None:
        return sentinel
    try:
        if pd.isna(value):
            return sentinel
    except (TypeError, ValueError):
        pass
    # int-valued floats (e.g. 5.0 from JSON numbers) share a bucket with ints.
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _normalize_categorical_nulls(
    df: pd.DataFrame, categorical_cols: list[str], sentinel: str = CATEGORICAL_NULL_SENTINEL
) -> pd.DataFrame:
    """Stringify categorical cells and collapse null variants to a common sentinel.

    Two representation drifts corrupt SDMetrics' TVComplement / ContingencySimilarity
    when left alone:

    * The LLM emits JSON ``null`` for what the real dataset stores as the literal
      string ``"None"``. SDMetrics drops the nulls and treats ``"None"`` as a
      regular category, so the bins never overlap.
    * The LLM parses JSON numbers (``5``, ``5.0``) into Python ``int`` / ``float``
      while the real column stays as ``str("5")`` (or vice versa). ``value_counts``
      hashes these to separate buckets, so even a perfect one-value distribution
      scores TVComplement ≈ 0.

    Normalizing both sides to consistent strings collapses both failure modes.
    """
    if not categorical_cols:
        return df
    present = [c for c in categorical_cols if c in df.columns]
    if not present:
        return df
    df = df.copy()
    for col in present:
        df[col] = df[col].map(lambda v: _normalize_categorical_value(v, sentinel))
    return df


def total_variation_distance(real: pd.Series, synthetic: pd.Series) -> float:
    """Compute total variation distance between two categorical distributions.

    TVD = 0.5 * sum(|P(x) - Q(x)|) for all categories x.
    Returns a value between 0 (identical) and 1 (completely different).
    """
    real_counts = real.value_counts(normalize=True, dropna=False)
    syn_counts = synthetic.value_counts(normalize=True, dropna=False)

    # Get all unique categories from both distributions
    all_categories = set(real_counts.index) | set(syn_counts.index)

    tvd = 0.0
    for category in all_categories:
        real_prob = real_counts.get(category, 0.0)
        syn_prob = syn_counts.get(category, 0.0)
        tvd += abs(real_prob - syn_prob)

    return float(tvd * 0.5)


def summarize_categorical_columns(
    real_df: pd.DataFrame, syn_df: pd.DataFrame, columns: list[str]
) -> list[dict[str, object]]:
    """Summarize categorical column distributions with full English metric names."""
    categorical_summary: list[dict[str, object]] = []
    for column in columns:
        if column not in real_df.columns or column not in syn_df.columns:
            continue

        real_counts = real_df[column].value_counts(normalize=True, dropna=False)
        syn_counts = syn_df[column].value_counts(normalize=True, dropna=False)

        # Compute TVD
        tvd = total_variation_distance(real_df[column], syn_df[column])

        # Full category universe (union of real and synthetic). Truncating to
        # real's top-10 hides synthetic-only categories and long-tail real gaps
        # that still contribute to TVD.
        real_freq = {str(k): float(v) for k, v in real_counts.items()}
        syn_freq = {str(k): float(v) for k, v in syn_counts.items()}
        all_keys = set(real_freq) | set(syn_freq)

        entry: dict[str, object] = {
            "column": column,
            "real_count": int(real_df[column].notna().sum()),
            "synthetic_count": int(syn_df[column].notna().sum()),
            "real_unique_values": int(real_df[column].nunique(dropna=False)),
            "synthetic_unique_values": int(syn_df[column].nunique(dropna=False)),
            "total_variation_distance": tvd,
            "top_categories_real": {k: real_freq.get(k, 0.0) for k in all_keys},
            "top_categories_synthetic": {k: syn_freq.get(k, 0.0) for k in all_keys},
        }

        # Category validity
        entry["category_adherence"] = CategoryAdherence.compute_breakdown(
            real_df[column], syn_df[column]
        )
        try:
            entry["category_coverage"] = CategoryCoverage.compute_breakdown(
                real_df[column], syn_df[column]
            )
        except Exception as e:
            logger.error(e)
            entry["category_coverage"] = {"score": 0.0}
        try:
            entry["tv_complement"] = TVComplement.compute_breakdown(real_df[column], syn_df[column])
        except Exception as e:
            logger.error(e)
            entry["tv_complement"] = {"score": 0.0}
        categorical_summary.append(entry)

    return categorical_summary


def _summarize_contingency_pairs(
    real_df: pd.DataFrame, syn_df: pd.DataFrame, columns: list[str]
) -> list[dict[str, object]]:
    """Summarize categorical column pair relationships (contingency)."""
    cat_cols_present = [c for c in columns if c in real_df.columns and c in syn_df.columns]
    contingency_summary = dict()
    cat_pairs_list = list(combinations(cat_cols_present, 2))
    for c1, c2 in cat_pairs_list:
        pair_key = f"{c1}__{c2}"
        breakdown = ContingencySimilarity.compute_breakdown(real_df[[c1, c2]], syn_df[[c1, c2]])
        contingency_summary[pair_key] = breakdown
        contingency_summary[pair_key]["key"] = pair_key
        contingency_summary[pair_key]["c1"] = c1
        contingency_summary[pair_key]["c2"] = c2
    return list(contingency_summary.values())


def load_metadata(dataset_name: str) -> dict[str, Any]:
    """Load SDMetrics metadata JSON directly from the metadata directory.

    Args:
        dataset_name: Name of the dataset metadata file stem.
    """
    assert dataset_name, "dataset_name must be a non-empty string"
    return load_metadata_dict(dataset_name)


def prepare_dataframe(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Select evaluation columns and coerce types for metric computation.

    Args:
        df: DataFrame to prepare
        dataset_name: Name of the dataset to use for column definitions (defaults to tourism_spending)
    """
    all_columns = _get_all_columns(dataset_name)
    available = [c for c in all_columns if c in df.columns]
    result = df[available].copy()
    for col in _get_categorical_columns(dataset_name):
        if col in result.columns:
            result[col] = result[col].astype(str)
    for col in _get_numeric_columns(dataset_name):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def _select_disclosure_columns(
    real_df: pd.DataFrame, syn_df: pd.DataFrame, schema: ResolvedSchema
) -> tuple[list[str], list[str]]:
    """Pick metadata-driven known/sensitive columns for disclosure protection."""
    shared_columns = set(real_df.columns) & set(syn_df.columns)

    known_candidates = schema.categorical
    known_present = [col for col in known_candidates if col in shared_columns]

    sensitive_candidates = schema.numerical
    sensitive_present = [col for col in sensitive_candidates if col in shared_columns]

    # Keep disclosure roles disjoint.
    sensitive_set = set(sensitive_present)
    known_present = [col for col in known_present if col not in sensitive_set]

    return known_present, sensitive_present


def correlation_gap(
    real_df: pd.DataFrame, syn_df: pd.DataFrame, columns: list[str]
) -> dict[str, object]:
    """Compute correlation matrix gap for numerical columns only.

    Reports real values, synthetic values, and gaps for all metrics including Frobenius norms.
    """
    if len(columns) < 2:
        return {
            "real_corr": {},
            "synthetic_corr": {},
            "abs_diff": {},
            "frobenius_norm_real": float("nan"),
            "frobenius_norm_synthetic": float("nan"),
            "frobenius_norm_gap": float("nan"),
            "max_abs_gap": float("nan"),
            "max_abs_gap_pair": None,
        }

    real_corr = real_df[columns].corr().fillna(0.0)
    syn_corr = syn_df[columns].corr().fillna(0.0)

    diff = real_corr - syn_corr

    abs_diff = diff.abs()
    max_idx = abs_diff.stack().idxmax() if not abs_diff.empty else (None, None)

    # Compute Frobenius norms for both matrices
    real_frobenius = float(np.linalg.norm(real_corr.to_numpy()))
    syn_frobenius = float(np.linalg.norm(syn_corr.to_numpy()))
    frobenius_gap = float(np.linalg.norm(diff.to_numpy()))

    res = {
        "real_corr": real_corr.round(4).to_dict(),
        "synthetic_corr": syn_corr.round(4).to_dict(),
        "abs_diff": abs_diff.round(4).to_dict(),
        "frobenius_norm_real": real_frobenius,
        "frobenius_norm_synthetic": syn_frobenius,
        "frobenius_norm_gap": frobenius_gap,
        "max_abs_gap": float(abs_diff.to_numpy().max() if not abs_diff.empty else float("nan")),
        "max_abs_gap_pair": list(max_idx) if max_idx else None,
    }
    return res


def _has_min_pairwise_correlation_samples(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    c1: str,
    c2: str,
) -> bool:
    """Return whether both datasets have enough aligned rows for correlation on one pair."""
    real_pair = real_df[[c1, c2]].dropna()
    syn_pair = syn_df[[c1, c2]].dropna()
    return len(real_pair) >= 2 and len(syn_pair) >= 2


@_timed
def evaluate_datasets(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema: ResolvedSchema,
    *,
    include_privacy_and_utility: bool = True,
) -> dict[str, object]:
    """Evaluate synthetic data quality for both numerical and categorical columns.

    Combines custom metrics with SDMetrics library for comprehensive evaluation including:
    - Statistical fidelity (distributions, correlations, SVD)
    - Privacy metrics (novelty, DCR protection, disclosure)
    - Utility metrics (ML efficacy, detection)
    - Validity metrics (boundary adherence, coverage)

    Args:
        real_df: Real/target dataset
        syn_df: Synthetic dataset
        dataset_metadata: Metadata dict or path to metadata file

    Returns:
        Dictionary containing:
        - numerical_columns: list of numerical column names
        - categorical_columns: list of categorical column names
        - numerical_summary: per-column statistics for numerical columns
        - categorical_summary: per-column statistics for categorical columns
        - correlation: correlation matrix gap (numerical columns only)
        - svd: SVD-based metrics for data structure comparison (numerical columns only)
        - category_metrics: unified metrics dictionary for backward compatibility
        - columns: all evaluated columns (backward compatibility)
        - column_summary: combined summary (backward compatibility)
        - sdmetrics: comprehensive SDMetrics-based evaluation (privacy, utility, validity)
    """
    assert isinstance(real_df, pd.DataFrame), "real_df must be a pandas DataFrame"
    assert isinstance(syn_df, pd.DataFrame), "syn_df must be a pandas DataFrame"
    assert not real_df.empty, "real_df must not be empty"
    assert not syn_df.empty, "syn_df must not be empty"

    shared_metadata_columns = [
        column
        for column in schema.columns
        if column in real_df.columns and column in syn_df.columns
    ]
    assert shared_metadata_columns, "No metadata columns are present in both real_df and syn_df"
    real_df = real_df.loc[:, shared_metadata_columns].copy()
    syn_df = syn_df.loc[:, shared_metadata_columns].copy()

    # Collapse null-like values in categorical columns to a single string so the
    # LLM's JSON `null` and the real data's "None" string land in the same bin.
    real_df = _normalize_categorical_nulls(real_df, schema.categorical)
    syn_df = _normalize_categorical_nulls(syn_df, schema.categorical)

    # Evaluate numerical columns with consolidated metrics
    numerical_summary, correlation_summary = summarize_numerical_columns(
        real_df, syn_df, schema.numerical
    )
    numerical_scores = [item["ks_complement"]["score"] for item in numerical_summary]

    # Evaluate categorical columns with consolidated metrics
    categorical_summary = summarize_categorical_columns(
        real_df, syn_df, schema.categorical
    )
    contingency_summary = _summarize_contingency_pairs(
        real_df, syn_df, schema.categorical
    )
    categorical_scores = [item["tv_complement"]["score"] for item in categorical_summary]

    # Privacy and utility metrics. NewRowSynthesis (~40s) and
    # DCRBaselineProtection (~400s) are unconditionally skipped — their cost
    # makes them unusable inside the comb-repair loop and no segment-tree
    # consumer reads them. Disclosure and LogisticDetection are gated by the
    # caller flag so non-comb callers can still request them.
    privacy_metrics: dict[str, Any] = {}
    utility_metrics: dict[str, Any] = {}
    if include_privacy_and_utility:
        # Disclosure protection
        known_present, sensitive_present = _select_disclosure_columns(real_df, syn_df, schema)
        if known_present and sensitive_present:
            privacy_metrics["disclosure_protection"] = DisclosureProtection.compute(
                real_df, syn_df, known_present, sensitive_present
            )

        # Normalize "None" strings to NaN before detection to avoid unknown-category errors
        none_marker = "None"
        real_detect = real_df.copy()
        syn_detect = syn_df.copy()
        for col in real_detect.columns:
            if real_detect[col].dtype == object:
                real_detect[col] = real_detect[col].replace(none_marker, np.nan)
                syn_detect[col] = syn_detect[col].replace(none_marker, np.nan)
        try:
            utility_metrics["logistic_detection_score"] = LogisticDetection.compute(
                real_detect, syn_detect, schema.metadata_dict
            )
        except Exception as e:
            logger.warning("logistic_detection_failed error=%s", e)
            utility_metrics["logistic_detection_score"] = None

    # Compute SDMetrics overall quality score
    with _timer("QualityReport"):
        report = QualityReport()
        report.generate(real_df, syn_df, schema.metadata_dict, verbose=False)
        column_shapes = report.get_details(property_name='Column Shapes')
        column_pair_trends = report.get_details(property_name='Column Pair Trends')
    correlation_scores = [item["score"] for item in correlation_summary["details"]]
    contingency_scores = [item["score"] for item in contingency_summary]

    # Convert to queryable dicts for MongoDB storage
    # Handle DataFrames by converting to list of dicts
    if isinstance(column_shapes, pd.DataFrame):
        column_shapes = column_shapes.to_dict('records')
    if isinstance(column_pair_trends, pd.DataFrame):
        column_pair_trends = column_pair_trends.to_dict('records')

    # column_shapes: {column_name: {"score": float, "metric": str}}
    column_shapes_dict = {}
    for item in column_shapes:
        col_name = str(item["Column"])
        column_shapes_dict[col_name] = {
            "score": float(item["Score"]),
            "metric": str(item["Metric"]),
        }

    # column_pair_trends: {pair_key: {"score": float, "metric": str}}
    def _coerce_score(raw: object) -> float | None:
        # SDV returns NaN when CorrelationSimilarity / ContingencySimilarity
        # can't be computed (constant column, too few non-null rows, etc.).
        # Surface as None so downstream consumers can distinguish
        # "score unavailable" from "score = 0".
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(value) else value

    column_shapes_dict = {
        col_name: {"score": _coerce_score(entry["score"]), "metric": entry["metric"]}
        for col_name, entry in column_shapes_dict.items()
    }

    column_pair_trends_dict = {}
    for item in column_pair_trends:
        col_x = str(item["Column 1"])
        col_y = str(item["Column 2"])
        pair_key = f"{col_x} vs {col_y}"
        column_pair_trends_dict[pair_key] = {
            "score": _coerce_score(item["Score"]),
            "metric": str(item["Metric"]),
        }

    # Build return lists for column_pair_trends with column names
    column_pair_trends_list = []
    for item in column_pair_trends:
        col_x = str(item["Column 1"])
        col_y = str(item["Column 2"])
        column_pair_trends_list.append({
            "column1": col_x,
            "column2": col_y,
            "score": _coerce_score(item["Score"]),
            "metric": str(item["Metric"]),
        })

    # Extract correlation pairs from correlation_summary details
    correlation_pairs_dict = {}
    if "details" in correlation_summary:
        for pair_data in correlation_summary["details"]:
            pair_key = pair_data.get("pair_key", "unknown")
            correlation_pairs_dict[pair_key] = {
                "score": pair_data.get("score", np.nan)
            }

    # Build correlation dict without details
    correlation_dict = {
        k: v for k, v in correlation_summary.items() if k != "details"
    }

    result = {
        "baseline_rows": len(real_df),
        "synthetic_rows": len(syn_df),
        "overall_quality_score": report.get_score(),
        "numerical_summary": numerical_summary,
        "numerical_score": np.mean(numerical_scores) if numerical_scores else 0,
        "categorical_summary": categorical_summary,
        "categorical_score": np.mean(categorical_scores) if categorical_scores else 0.0,
        "correlation_summary": correlation_summary,
        "correlation": correlation_dict,
        "correlation_score": np.mean(correlation_scores) if correlation_scores else 0.0,
        "correlation_pairs": correlation_pairs_dict,
        "contingency_summary": contingency_summary,
        "contingency_score": np.mean(contingency_scores) if contingency_scores else 0,
        "privacy": privacy_metrics,
        "utility": utility_metrics,
        "column_pair_trends": column_pair_trends_list,
        "column_shapes": column_shapes_dict,
    }

    return _convert_numpy_types(result)
