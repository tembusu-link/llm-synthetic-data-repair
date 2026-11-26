"""Gap detection, directives, and convergence checks."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

from .core import GapDirective, GapMetric, NumericTextureProfile
from .schema import ResolvedSchema
from .stats import _scatter_recipe, _scatter_recipe_text

logger = logging.getLogger(__name__)

DIRECTIVE_MIN_GAP_BY_TYPE: dict[str, float] = {
    "sentinel_gap": 0.10,
    "quantile_gap": 0.10,
    "categorical_gap": 0.10,
    "corr_gap": 0.10,
    "contingency_gap": 0.10,
}
# Score-based gating: emit directives for any column/pair whose SDMetrics
# quality score is below this ceiling (bad enough to need fixing); skip
# otherwise (column is already good, don't nitpick). When the score is
# unavailable (None — SDV couldn't compute), fall back to the per-cell
# gap-magnitude table above.
DIRECTIVE_SCORE_CEILING = 0.90
# Debug-directive score ceiling: wider net for post-hoc inspection.
DEBUG_DIRECTIVE_SCORE_CEILING = 0.90
DIRECTIVE_THRESHOLD_EPSILON = 1e-9
# Max per-column cells in a collapsed categorical_gap directive (top by |gap|).
CATEGORICAL_GAP_MAX_CELLS = 5


@contextmanager
def timer(label: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[TIMING] {label}: {elapsed:.3f}s")


def timed(func: Callable) -> Callable:
    """Decorator to time function execution."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"[TIMING] {func.__name__}: {elapsed:.3f}s")
        return result

    return wrapper


def _ensure_min_rows(df: pd.DataFrame, min_rows: int = 3) -> pd.DataFrame:
    """Fail fast when dataframe is too small for downstream SDMetrics computations."""
    assert len(df) >= min_rows, f"Dataframe must have at least {min_rows} rows"
    return df


def _extract_missing_gap_statistics(report: dict[str, object]) -> list[str]:
    missing: list[str] = []
    if "numerical_summary" not in report:
        missing.append("numerical_summary")
    if "categorical_summary" not in report:
        missing.append("categorical_summary")
    corr = report.get("correlation")
    if not isinstance(corr, dict):
        missing.append("correlation")
    else:
        for key in ("abs_diff", "real_corr", "synthetic_corr", "frobenius_norm_gap"):
            if key not in corr:
                missing.append(f"correlation.{key}")
    return missing


def _directive_gap_magnitude(directive: GapDirective) -> float | None:
    # Collapsed quantile_gap: mean relative drift across all cells.
    if directive.gap_type == "quantile_gap" and directive.cells:
        deltas: list[float] = []
        for cell in directive.cells:
            target = cell.get("target")
            current = cell.get("current")
            if target is None or current is None:
                continue
            try:
                t, c = float(target), float(current)
            except (TypeError, ValueError):
                continue
            deltas.append(abs(t - c) / max(abs(t), 1.0))
        return sum(deltas) / len(deltas) if deltas else None

    # Collapsed categorical_gap: max |target - current| across cells.
    # Using max (not mean) so a column with one severe value drift outranks
    # a column with many small ones.
    if directive.gap_type == "categorical_gap" and directive.cells:
        deltas: list[float] = []
        for cell in directive.cells:
            target = cell.get("target")
            current = cell.get("current")
            if target is None or current is None:
                continue
            try:
                t, c = float(target), float(current)
            except (TypeError, ValueError):
                continue
            deltas.append(abs(t - c))
        return max(deltas) if deltas else None

    if directive.target_value is None or directive.current_value is None:
        return None
    try:
        target_value = float(directive.target_value)
        current_value = float(directive.current_value)
    except (TypeError, ValueError):
        return None
    if directive.gap_type == "quantile_gap":
        return abs(target_value - current_value) / max(abs(target_value), 1.0)
    return abs(target_value - current_value)


def _exceeds_threshold(magnitude: float, threshold: float) -> bool:
    return magnitude > threshold + DIRECTIVE_THRESHOLD_EPSILON


def _build_score_lookups(
    report: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Return (column_shapes_dict, column_pair_trends_dict) keyed for directive lookup.

    SDV's ``Column Pair Trends`` stores each pair in a single direction; the
    directive-side lookup may query the reverse order (schema-iteration vs
    SDV-internal order). Index both directions so the lookup is
    order-insensitive.
    """
    column_shapes_dict = report.get("column_shapes", {}) or {}

    column_pair_trends_list = report.get("column_pair_trends", [])
    column_pair_trends_dict: dict[str, object] = {}
    if isinstance(column_pair_trends_list, list):
        for item in column_pair_trends_list:
            col1 = item.get("column1") or ""
            col2 = item.get("column2") or ""
            if col1 and col2:
                score = item.get("score", 1.0)
                column_pair_trends_dict[f"{col1} vs {col2}"] = score
                column_pair_trends_dict[f"{col2} vs {col1}"] = score
    elif isinstance(column_pair_trends_list, dict):
        # Pre-built dict (e.g. from tests). Augment with reverse keys.
        column_pair_trends_dict = dict(column_pair_trends_list)
        for key, value in list(column_pair_trends_list.items()):
            if " vs " in key:
                a, b = key.split(" vs ", 1)
                column_pair_trends_dict.setdefault(f"{b} vs {a}", value)

    return column_shapes_dict, column_pair_trends_dict


def _lookup_sdmetrics_score(
    directive: GapDirective | dict,
    column_shapes_dict: dict[str, object],
    column_pair_trends_dict: dict[str, object],
) -> float | None:
    """Extract SDMetrics column/pair quality score for a directive.

    Returns None when the score is missing or NaN (SDV uses NaN to signal
    "metric undefined" — e.g. constant column, too few non-null rows).
    """
    gap_type = (
        directive.gap_type if isinstance(directive, GapDirective) else directive.get("gap_type")
    )
    column = directive.column if isinstance(directive, GapDirective) else directive.get("column")

    def _finite_float(raw: object) -> float | None:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return None if value != value else value  # NaN check without math import

    def _unwrap(entry: object) -> object:
        # Entries can be a flat number, a nested {"score": ...} dict, or None
        # (NaN scores are coerced to None upstream). Normalize to the raw value
        # _finite_float will vet.
        if entry is None:
            return None
        if isinstance(entry, (int, float)):
            return entry
        if isinstance(entry, dict):
            return entry.get("score")
        return None

    if gap_type in ("sentinel_gap", "quantile_gap", "categorical_gap"):
        col_name = str(column) if isinstance(column, str) else column
        if col_name in column_shapes_dict:
            return _finite_float(_unwrap(column_shapes_dict.get(col_name)))
    elif gap_type in ("corr_gap", "contingency_gap"):
        col_pair = column if isinstance(column, list) else (column, column)
        pair_key = f"{col_pair[0]} vs {col_pair[1]}"
        if pair_key in column_pair_trends_dict:
            return _finite_float(_unwrap(column_pair_trends_dict.get(pair_key)))
    return None


def _lookup_sdmetric_name(
    directive: GapDirective | dict,
    column_shapes_dict: dict[str, object],
    column_pair_trends_dict: dict[str, object],
) -> str | None:
    """Return the SDMetrics metric name (e.g. 'TVComplement') backing a directive's score."""
    gap_type = (
        directive.gap_type if isinstance(directive, GapDirective) else directive.get("gap_type")
    )
    column = directive.column if isinstance(directive, GapDirective) else directive.get("column")

    def _extract(entry: object) -> str | None:
        if isinstance(entry, dict):
            metric = entry.get("metric")
            return str(metric) if metric is not None else None
        return None

    if gap_type in ("sentinel_gap", "quantile_gap", "categorical_gap"):
        if isinstance(column, str) and column in column_shapes_dict:
            return _extract(column_shapes_dict.get(column))
    elif gap_type in ("corr_gap", "contingency_gap"):
        col_pair = column if isinstance(column, list) else (column, column)
        pair_key = f"{col_pair[0]} vs {col_pair[1]}"
        if pair_key in column_pair_trends_dict:
            return _extract(column_pair_trends_dict.get(pair_key))
    return None


def _passes_directive_threshold(directive: GapDirective) -> bool:
    # With score-based gating at generation time, every emitted directive
    # already passed the column/pair quality filter. The static-threshold
    # post-filter would only drop low-magnitude cells we deliberately kept.
    return True


def _should_emit_for_score(
    score: float | None, gap_magnitude: float, fallback_threshold: float
) -> bool:
    """Primary gate: column/pair SDMetrics score.

    - ``score < DIRECTIVE_SCORE_CEILING`` → emit (quality is bad, every cell
      of the column/pair is fair game for the LLM to address).
    - ``score >= DIRECTIVE_SCORE_CEILING`` → skip (quality is good; don't
      nitpick individual cells).

    Fallback when the score is ``None`` (SDV couldn't compute): apply the
    per-cell ``fallback_threshold`` from :data:`DIRECTIVE_MIN_GAP_BY_TYPE`.
    """
    if score is not None:
        return score < DIRECTIVE_SCORE_CEILING
    return _exceeds_threshold(gap_magnitude, fallback_threshold)


def _build_quantile_gap_directive(
    col: str,
    target_profile: NumericTextureProfile,
    current_profile: NumericTextureProfile,
    col_score: float | None,
) -> GapDirective | None:
    """Collapse six per-percentile quantile gaps into one directive per column.

    KSComplement is a single column-level score, so emitting six separate
    quantile directives is wasteful of the ``max_directives`` budget and all
    share the same ``score_before``. Pack the percentiles as cells instead.
    Returns None if no usable percentile data.
    """
    cells: list[dict] = []
    deltas: list[float] = []
    for q_name in ("p10", "p25", "p50", "p75", "p90", "p95"):
        target_q = target_profile.quantiles.get(q_name)
        current_q = current_profile.quantiles.get(q_name)
        if target_q is None or current_q is None:
            continue
        target_q = float(target_q)
        current_q = float(current_q)
        cells.append(
            {"metric": q_name, "target": round(target_q, 3), "current": round(current_q, 3)}
        )
        deltas.append(target_q - current_q)
    if not cells:
        return None
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    if pos and not neg:
        overall_direction = "increase"
    elif neg and not pos:
        overall_direction = "decrease"
    else:
        overall_direction = "mixed"
    note = (
        f"KSComplement={col_score:.3f}; {len(cells)} percentiles tracked"
        if col_score is not None
        else f"{len(cells)} percentiles tracked"
    )
    return GapDirective(
        priority=0,
        gap_type="quantile_gap",
        column=col,
        metric="shape",
        direction=overall_direction,
        target_value=None,
        current_value=None,
        note=note,
        score_before=col_score,
        cells=cells,
    )


def _build_categorical_gap_directive(
    col: str,
    target_top: dict[str, float],
    current_top: dict[str, float],
    col_score: float | None,
    cell_threshold: float,
    max_cells: int,
) -> GapDirective | None:
    """Collapse per-category frequency gaps into one directive per column.

    Mirrors the quantile_gap pattern: one directive per low-quality column,
    with the top ``max_cells`` categories (by |target - current|) above
    ``cell_threshold`` packed as cells. Returns None when nothing crosses
    the threshold.
    """
    all_cats = set(target_top) | set(current_top)
    scored: list[tuple[str, float, float, float]] = []
    for cat in all_cats:
        target_freq = float(target_top.get(cat, 0.0))
        current_freq = float(current_top.get(cat, 0.0))
        diff = target_freq - current_freq
        if abs(diff) < cell_threshold - DIRECTIVE_THRESHOLD_EPSILON:
            continue
        scored.append((cat, target_freq, current_freq, diff))
    if not scored:
        return None
    scored.sort(key=lambda x: abs(x[3]), reverse=True)
    scored = scored[:max_cells]
    cells = [
        {"metric": cat, "target": round(t, 3), "current": round(c, 3)}
        for cat, t, c, _ in scored
    ]
    deltas = [d for _, _, _, d in scored]
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    if pos and not neg:
        overall_direction = "increase"
    elif neg and not pos:
        overall_direction = "decrease"
    else:
        overall_direction = "mixed"
    note = (
        f"TVComplement={col_score:.3f}; {len(cells)} categories off"
        if col_score is not None
        else f"{len(cells)} categories off"
    )
    return GapDirective(
        priority=0,
        gap_type="categorical_gap",
        column=col,
        metric="marginal",
        direction=overall_direction,
        target_value=None,
        current_value=None,
        note=note,
        score_before=col_score,
        cells=cells,
    )


def _extract_pair_score(
    pair_key: str, column_pair_trends_dict: dict[str, object]
) -> float | None:
    """Resolve a column-pair SDMetrics score to a finite float, or None."""
    entry = column_pair_trends_dict.get(pair_key)
    if entry is None:
        return None
    if isinstance(entry, (int, float)):
        value = float(entry)
    elif isinstance(entry, dict):
        raw = entry.get("score")
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
    else:
        return None
    return None if value != value else value


@timed
def compute_gap_directives(
    target_profiles: dict[str, NumericTextureProfile],
    current_profiles: dict[str, NumericTextureProfile],
    target_df: pd.DataFrame,
    current_df: pd.DataFrame,
    dataset_metadata: dict[str, Any],
    schema: ResolvedSchema,
    target_contingency_pairs: list[dict[str, object]] | None = None,
    current_contingency_pairs: list[dict[str, object]] | None = None,
    max_directives: int = 8,
    report: dict[str, object] | None = None,
) -> list[GapDirective]:
    """
    Compute prioritized GapDirectives by comparing target and current profiles.

    This function identifies the most significant gaps between synthetic and
    target data, expressed in a universal, dataset-agnostic language.

    Directives are ranked by:
    1. Gap type (semantic importance)
    2. Gap magnitude (how far from target)
    3. Column/pair quality score (lower = higher priority for improvement)

    Args:
        target_profiles: Target NumericTextureProfiles.
        current_profiles: Current synthetic NumericTextureProfiles.
        target_df: Target DataFrame for categorical/correlation gaps.
        current_df: Current synthetic DataFrame.
        cat_cols: Categorical column names.
        num_cols: Numeric column names.
        max_directives: Maximum number of directives to return.

    Returns:
        List of GapDirective objects, sorted by priority.
    """
    directives: list[GapDirective] = []

    # Extract SDMetrics quality scores for column-aware prioritization.
    # Callers that already have a fresh evaluate_datasets report (e.g. the
    # previous iteration's post-generation statistics) can pass it via `report`
    # to avoid recomputing the same thing.
    if report is None:
        from llm_segment_repair.metrics import evaluate_datasets

        report = evaluate_datasets(target_df, current_df, schema=schema)
    column_shapes_dict, column_pair_trends_dict = _build_score_lookups(report)

    # Sentinel mass gaps and quantile gaps.
    # Gating: column KSComplement. Below ceiling → emit every cell with a
    # nonzero gap; at/above ceiling → skip the column entirely.
    for col in schema.numerical:
        if col not in target_profiles or col not in current_profiles:
            continue

        target_p = target_profiles[col]
        current_p = current_profiles[col]

        col_score = _extract_column_score(str(col), column_shapes_dict)
        if col_score is not None and col_score >= DIRECTIVE_SCORE_CEILING:
            continue

        # Sentinel mass gaps
        target_sentinels = {s.value: s for s in (target_p.sentinels or [])}
        current_sentinels = {s.value: s for s in (current_p.sentinels or [])}
        for val in set(target_sentinels) | set(current_sentinels):
            target_mass = target_sentinels[val].mass if val in target_sentinels else 0.0
            current_mass = current_sentinels[val].mass if val in current_sentinels else 0.0
            mass_gap = abs(target_mass - current_mass)
            if _should_emit_for_score(
                col_score, mass_gap, DIRECTIVE_MIN_GAP_BY_TYPE["sentinel_gap"]
            ):
                direction = "increase" if current_mass < target_mass else "decrease"
                directives.append(
                    GapDirective(
                        priority=0,
                        gap_type="sentinel_gap",
                        column=col,
                        metric=f"sentinel={val}",
                        direction=direction,
                        target_value=round(target_mass, 4),
                        current_value=round(current_mass, 4),
                        note=f"Sentinel {val} mass off by {mass_gap:.1%}",
                    )
                )

        # Quantile gaps: one collapsed directive per column with every
        # percentile packed as a cell. Skip the column if no percentile has
        # both a target and a current value.
        #
        # When the column score is unavailable (None), fall back to the old
        # per-cell threshold: emit the directive only if at least one
        # percentile has a relative gap above the threshold.
        emit_quantile = col_score is not None
        if not emit_quantile:
            for q_name in ("p10", "p25", "p50", "p75", "p90", "p95"):
                target_q = target_p.quantiles.get(q_name)
                current_q = current_p.quantiles.get(q_name)
                if target_q is None or current_q is None or target_q == 0:
                    continue
                rel_gap = abs(float(target_q) - float(current_q)) / abs(float(target_q))
                if _exceeds_threshold(rel_gap, DIRECTIVE_MIN_GAP_BY_TYPE["quantile_gap"]):
                    emit_quantile = True
                    break
        if emit_quantile:
            qd = _build_quantile_gap_directive(col, target_p, current_p, col_score)
            if qd is not None:
                directives.append(qd)

    missing_stats = _extract_missing_gap_statistics(report)
    if missing_stats:
        logger.warning(
            "missing_gap_statistics kind=directives missing=%s",
            ",".join(missing_stats),
        )

    # Categorical frequency gaps — one collapsed directive per column, with the
    # top-N largest per-category |gap| values packed as cells. Mirrors the
    # quantile_gap pattern so a single low-quality column can't flood the
    # directive budget.
    for entry in report.get("categorical_summary", []):
        col = entry.get("column")
        if col not in schema.categorical:
            continue
        target_top = entry.get("top_categories_real", {}) or {}
        current_top = entry.get("top_categories_synthetic", {}) or {}
        all_cats = set(target_top) | set(current_top)
        if len(all_cats) <= 1:
            continue
        col_score = _extract_column_score(str(col), column_shapes_dict)
        if col_score is not None and col_score >= DIRECTIVE_SCORE_CEILING:
            continue
        cd = _build_categorical_gap_directive(
            str(col),
            target_top,
            current_top,
            col_score,
            cell_threshold=DIRECTIVE_MIN_GAP_BY_TYPE["categorical_gap"],
            max_cells=CATEGORICAL_GAP_MAX_CELLS,
        )
        if cd is not None:
            directives.append(cd)

    # Correlation gaps (derived from evaluate_datasets correlation matrices).
    # Gating: pair CorrelationSimilarity. Below ceiling → emit; at/above → skip.
    corr = report.get("correlation", {})
    real_corr = corr.get("real_corr", {}) if isinstance(corr, dict) else {}
    synthetic_corr = corr.get("synthetic_corr", {}) if isinstance(corr, dict) else {}
    for i, col_a in enumerate(schema.numerical):
        for col_b in schema.numerical[i + 1 :]:
            target_corr = (real_corr.get(col_a) or {}).get(col_b)
            current_corr = (synthetic_corr.get(col_a) or {}).get(col_b)
            if target_corr is None or current_corr is None:
                continue
            target_val = float(target_corr)
            current_val = float(current_corr)
            corr_gap = abs(target_val - current_val)
            pair_score = _extract_pair_score(
                f"{col_a} vs {col_b}", column_pair_trends_dict
            )
            if pair_score is not None and pair_score >= DIRECTIVE_SCORE_CEILING:
                continue
            if _should_emit_for_score(
                pair_score, corr_gap, DIRECTIVE_MIN_GAP_BY_TYPE["corr_gap"]
            ):
                direction = "increase" if current_val < target_val else "decrease"
                directives.append(
                    GapDirective(
                        priority=0,
                        gap_type="corr_gap",
                        column=[col_a, col_b],
                        metric="pearson_r",
                        direction=direction,
                        target_value=round(target_val, 3),
                        current_value=round(current_val, 3),
                        note=f"Correlation gap: {current_val:.2f} vs target {target_val:.2f}",
                    )
                )

    # Contingency similarity gaps are intentionally not emitted — the LLM
    # struggles to act on joint-distribution directives, and they tended to
    # crowd out higher-signal column-level gaps in the directive budget.

    directives = [d for d in directives if _passes_directive_threshold(d)]

    for d in directives:
        d.score_before = _lookup_sdmetrics_score(d, column_shapes_dict, column_pair_trends_dict)

    # Global score-based selection: rank every directive by column/pair quality
    # score ascending (worst first), tiebreak by gap magnitude descending
    # (largest drift first). Take the top `max_directives`. No round-robin
    # stratification — the prompt should always focus on the biggest quality
    # drops regardless of directive type.
    def _priority_key(d: GapDirective) -> tuple[float, float]:
        score = d.score_before if d.score_before is not None else 1.0
        magnitude = _directive_gap_magnitude(d) or 0.0
        return (score, -magnitude)

    directives.sort(key=_priority_key)
    selected = directives[:max_directives]
    selected_ids = {id(d) for d in selected}

    # Assign priorities based on final selection order
    for i, d in enumerate(selected):
        d.priority = i + 1

    # Full ranked list, including directives that fell below the max_directives
    # cutoff. Marked with '*' when selected for the prompt, '-' when dropped.
    logger.info("gap_directives_full_ranked total=%d max_directives=%d", len(directives), max_directives)
    for rank, d in enumerate(directives, start=1):
        col_str = d.column if isinstance(d.column, str) else f"[{', '.join(d.column)}]"
        quality = f"{d.score_before:.3f}" if d.score_before is not None else "n/a"
        magnitude = _directive_gap_magnitude(d)
        magnitude_str = f"{magnitude:.4f}" if magnitude is not None else "n/a"
        sdmetric = _lookup_sdmetric_name(d, column_shapes_dict, column_pair_trends_dict)
        flag = "*" if id(d) in selected_ids else "-"
        logger.info(
            "  %s gap_directive[rank=%d] %s %s %s sdmetric=%s quality=%s |gap|=%s target=%s current=%s",
            flag,
            rank,
            d.gap_type,
            col_str,
            d.metric,
            sdmetric or "n/a",
            quality,
            magnitude_str,
            d.target_value,
            d.current_value,
        )

    selected_by_type: dict[str, int] = {}
    for d in selected:
        selected_by_type[d.gap_type] = selected_by_type.get(d.gap_type, 0) + 1
    logger.info(
        "gap_directives_selected total=%d by_type=%s",
        len(selected),
        selected_by_type,
    )
    return selected


def _extract_column_score(col_name: str, column_shapes_dict: dict[str, object]) -> float | None:
    """Resolve the per-column SDMetrics score to a float, or None if unavailable."""
    entry = column_shapes_dict.get(str(col_name))
    if entry is None:
        return None
    if isinstance(entry, (int, float)):
        return float(entry)
    raw = entry.get("score") if isinstance(entry, dict) else None
    return float(raw) if raw is not None else None


def compute_debug_gap_directives(
    report: dict[str, object],
    schema: ResolvedSchema,
    target_contingency_pairs: list[dict[str, object]] | None = None,
    score_threshold: float = DEBUG_DIRECTIVE_SCORE_CEILING,
    target_profiles: dict[str, NumericTextureProfile] | None = None,
    current_profiles: dict[str, NumericTextureProfile] | None = None,
) -> list[GapDirective]:
    """Emit every potential directive for columns/pairs whose SDMetrics score ≤ threshold.

    Bypasses gap-magnitude thresholds, per-column skip rules, and `max_directives`
    truncation. Intended for persistence to iteration_metadata so low-quality
    columns can be inspected post-hoc; these are NOT formatted into the LLM prompt.

    Coverage: categorical_gap (per-category), corr_gap (numerical pairs),
    contingency_gap (categorical pairs), and — when `target_profiles` /
    `current_profiles` are provided — quantile_gap (per-percentile) for
    numerical columns with bad KSComplement scores. The profiles come from
    `compute_all_texture_profiles` upstream, so the quantiles are already
    sentinel-aware and don't need recomputing.
    """
    directives: list[GapDirective] = []
    column_shapes_dict, column_pair_trends_dict = _build_score_lookups(report)

    # Categorical: per-category directives for every low-quality categorical column.
    for entry in report.get("categorical_summary", []) or []:
        col = entry.get("column")
        if col not in schema.categorical:
            continue
        col_score = _extract_column_score(str(col), column_shapes_dict)
        if col_score is None or col_score > score_threshold:
            continue
        target_top = entry.get("top_categories_real", {}) or {}
        current_top = entry.get("top_categories_synthetic", {}) or {}
        all_cats = set(target_top) | set(current_top)
        if len(all_cats) <= 1:
            continue
        for cat in all_cats:
            target_freq = float(target_top.get(cat, 0.0))
            current_freq = float(current_top.get(cat, 0.0))
            gap = target_freq - current_freq
            direction = "increase" if gap >= 0 else "decrease"
            directives.append(
                GapDirective(
                    priority=0,
                    gap_type="categorical_gap",
                    column=str(col),
                    metric=f"'{cat}' frequency",
                    direction=direction,
                    target_value=round(target_freq, 4),
                    current_value=round(current_freq, 4),
                    note=f"Category '{cat}' delta {abs(gap):.1%}",
                    score_before=col_score,
                )
            )

    # Correlation: every numerical pair with low column_pair_trends score.
    corr = report.get("correlation", {}) or {}
    real_corr = corr.get("real_corr", {}) if isinstance(corr, dict) else {}
    synthetic_corr = corr.get("synthetic_corr", {}) if isinstance(corr, dict) else {}
    for i, col_a in enumerate(schema.numerical):
        for col_b in schema.numerical[i + 1 :]:
            pair_key = f"{col_a} vs {col_b}"
            pair_entry = column_pair_trends_dict.get(pair_key)
            if pair_entry is None:
                continue
            pair_score = float(
                pair_entry if isinstance(pair_entry, (int, float)) else pair_entry.get("score", 1.0)
            )
            if pair_score > score_threshold:
                continue
            target_corr = (real_corr.get(col_a) or {}).get(col_b)
            current_corr = (synthetic_corr.get(col_a) or {}).get(col_b)
            if target_corr is None or current_corr is None:
                continue
            target_val = float(target_corr)
            current_val = float(current_corr)
            direction = "increase" if current_val < target_val else "decrease"
            directives.append(
                GapDirective(
                    priority=0,
                    gap_type="corr_gap",
                    column=[col_a, col_b],
                    metric="pearson_r",
                    direction=direction,
                    target_value=round(target_val, 3),
                    current_value=round(current_val, 3),
                    note=f"Correlation {current_val:.2f} vs target {target_val:.2f}",
                    score_before=pair_score,
                )
            )

    # Contingency: every categorical pair with low ContingencySimilarity, filtered
    # to pairs with meaningful real-world association.
    interesting_pairs: set[tuple[str, str]] = set()
    if target_contingency_pairs:
        for pair_entry in target_contingency_pairs:
            c1 = pair_entry.get("c1")
            c2 = pair_entry.get("c2")
            if c1 and c2:
                interesting_pairs.add((str(c1), str(c2)))
    contingency_sim_map: dict[tuple[str, str], float] = {}
    for entry in report.get("contingency_summary", []) or []:
        c1 = entry.get("c1")
        c2 = entry.get("c2")
        score = entry.get("score")
        if c1 and c2 and score is not None:
            contingency_sim_map[(str(c1), str(c2))] = float(score)
    for c1, c2 in interesting_pairs:
        sim_score = contingency_sim_map.get((c1, c2)) or contingency_sim_map.get((c2, c1))
        if sim_score is None or sim_score > score_threshold:
            continue
        directives.append(
            GapDirective(
                priority=0,
                gap_type="contingency_gap",
                column=[c1, c2],
                metric="contingency_similarity",
                direction="increase",
                target_value=1.0,
                current_value=round(float(sim_score), 3),
                note=f"Joint distribution similarity: {sim_score:.3f} vs target 1.000",
                score_before=float(sim_score),
            )
        )

    # Numerical: one collapsed quantile_gap per low-KSComplement column with
    # every percentile packed as a cell.
    if target_profiles and current_profiles:
        for col in schema.numerical:
            col_score = _extract_column_score(str(col), column_shapes_dict)
            if col_score is None or col_score > score_threshold:
                continue
            target_p = target_profiles.get(col)
            current_p = current_profiles.get(col)
            if target_p is None or current_p is None:
                continue
            qd = _build_quantile_gap_directive(str(col), target_p, current_p, col_score)
            if qd is not None:
                directives.append(qd)

    logger.info(
        "debug_gap_directives_generated total=%d score_threshold=%.2f",
        len(directives),
        score_threshold,
    )
    return directives


def format_gap_directives(
    directives: list[GapDirective],
    n_rows: int = 0,
    current_row_count: int = 0,
    target_total_rows: int = 0,
) -> str:
    """
    Format GapDirectives as structured text for the LLM prompt.

    Args:
        directives: List of GapDirective objects.
        n_rows: Batch size the LLM is about to generate.
        current_row_count: Rows already generated in prior batches of this
            dataset_segment. Combined with ``target_total_rows`` lets the
            instruction compute a deficit-aware count for this batch instead
            of repeating the per-batch target every time. Overshoot in early
            iterations is intentional — the LLM should close the overall gap
            rather than spreading correction across all iterations evenly.
        target_total_rows: Total rows the final synthetic dataset should
            have for this dataset_segment. When 0, instructions fall back to
            ``round(proportion * n_rows)``.

    Returns:
        Formatted string describing the gaps to address.
    """
    if not directives:
        return "No significant gaps detected."

    def _fmt(val: object) -> str:
        if val is None:
            return "n/a"
        if isinstance(val, float):
            return f"{val:.4g}"
        return str(val)

    def _fmt_pct(val: object, signed: bool = False) -> str:
        """Render a proportion / ratio as a percentage string."""
        if val is None:
            return "n/a"
        if isinstance(val, (int, float)):
            spec = "+.1f" if signed else ".1f"
            return f"{val * 100:{spec}}%"
        return str(val)

    # Per-gap-type semantics for cell/header values:
    # - "proportion": value lies in [0, 1] (frequency, p_zero, correlation) —
    #   render as percentage.
    # - "absolute":   value is an absolute quantity (e.g. quantile of mileage) —
    #   keep raw. Gap is still rendered as percentage because _directive_gap
    #   normalizes by |target| for quantile_gap.
    proportion_types = {"categorical_gap", "corr_gap", "sentinel_gap", "contingency_gap"}

    def _fmt_value(val: object, gap_type: str) -> str:
        if gap_type in proportion_types:
            return _fmt_pct(val)
        return _fmt(val)

    def _fmt_gap(val: object) -> str:
        # Gap is always a ratio / signed proportion → percentage.
        return _fmt_pct(val, signed=True)

    lines: list[str] = []
    for d in directives:
        col_str = d.column if isinstance(d.column, str) else f"[{', '.join(d.column)}]"
        quality = f"{d.score_before:.2f}" if d.score_before is not None else "n/a"
        header = (
            f"{d.priority}. [{d.gap_type}] {col_str} {d.metric} "
            f"(quality={quality})"
        )
        if d.cells:
            gap = _directive_gap(d.target_value, d.current_value, d.gap_type)
            lines.append(
                f"{header}: target={_fmt_value(d.target_value, d.gap_type)} "
                f"current={_fmt_value(d.current_value, d.gap_type)} gap={_fmt_gap(gap)}"
            )
            if d.gap_type == "contingency_gap":
                cols = d.column if isinstance(d.column, list) else [d.column]
                c1_name = str(cols[0]) if len(cols) > 0 else "c1"
                c2_name = str(cols[1]) if len(cols) > 1 else "c2"
                for cell in d.cells:
                    c1v = cell.get("c1_val")
                    c2v = cell.get("c2_val")
                    freq = cell.get("freq")
                    freq_str = _fmt_pct(freq)
                    lines.append(
                        f"   - {c1_name}={c1v} × {c2_name}={c2v} (target freq={freq_str})"
                    )
            else:
                for cell in d.cells:
                    metric = cell.get("metric", "")
                    t = cell.get("target")
                    c = cell.get("current")
                    cell_gap = _directive_gap(t, c, d.gap_type)
                    lines.append(
                        f"   - {metric}: target={_fmt_value(t, d.gap_type)} "
                        f"current={_fmt_value(c, d.gap_type)} gap={_fmt_gap(cell_gap)}"
                    )
        else:
            gap = _directive_gap(d.target_value, d.current_value, d.gap_type)
            lines.append(
                f"{header}: target={_fmt_value(d.target_value, d.gap_type)} "
                f"current={_fmt_value(d.current_value, d.gap_type)} gap={_fmt_gap(gap)}"
            )

    return "\n".join(lines)


def _directive_gap(target_val: object, current_val: object, gap_type: str) -> float | None:
    """Signed gap = target - current.

    Positive means the current synthetic under-shoots the target (need to
    increase); negative means it over-shoots (need to decrease). For
    quantile_gap, the result is normalized by |target|.
    """
    if target_val is None or current_val is None:
        return None
    try:
        t, c = float(target_val), float(current_val)
    except (ValueError, TypeError):
        return None
    diff = t - c
    if gap_type == "quantile_gap":
        return round(diff / max(abs(t), 1e-9), 4)
    return round(diff, 4)


def _deficit_count(
    target_proportion: float,
    current_proportion: float | None,
    n_rows: int,
    current_row_count: int,
    target_total_rows: int,
) -> int | None:
    """Records this batch should generate to close the overall-target gap.

    target_total = target_proportion * target_total_rows
    have_now     = current_proportion * current_row_count
    deficit      = max(0, target_total - have_now)
    # Cap at batch size; any remainder carries to future iterations.
    result       = min(deficit, n_rows)

    Returns None when target_total_rows is unknown (caller falls back to the
    per-batch proportion).
    """
    if target_total_rows <= 0 or n_rows <= 0:
        return None
    target_total = target_proportion * target_total_rows
    have_now = (current_proportion or 0.0) * max(current_row_count, 0)
    deficit = max(0.0, target_total - have_now)
    return int(round(min(deficit, n_rows)))


def _proportion_count_text(
    target_proportion: float,
    current_proportion: float | None,
    n_rows: int,
    current_row_count: int,
    target_total_rows: int,
) -> str:
    """Render a record-count string using deficit math when possible."""
    deficit = _deficit_count(
        target_proportion, current_proportion, n_rows, current_row_count, target_total_rows
    )
    if deficit is not None:
        return f"{deficit} records"
    if n_rows > 0:
        return f"{round(target_proportion * n_rows)} records"
    return f"~{target_proportion * 100:.0f}% of records"


def _format_directive_instruction(
    d: GapDirective,
    n_rows: int = 0,
    current_row_count: int = 0,
    target_total_rows: int = 0,
) -> str:
    """Return the actionable instruction text for a single directive."""
    if d.gap_type == "quantile_gap" and d.cells:
        # Collapsed form: one directive per column. Identify which intervals
        # need more or fewer records based on which quantiles are too high/low.
        gaps_by_percentile: list[tuple[str, float, float, float]] = []
        for cell in d.cells:
            metric = cell.get("metric", "")
            target = cell.get("target")
            current = cell.get("current")
            if target is None or current is None:
                continue
            try:
                t, c = float(target), float(current)
                gap = t - c  # gap = target - current
                gaps_by_percentile.append((metric, t, c, gap))
            except (TypeError, ValueError):
                continue

        if not gaps_by_percentile:
            return f"Match target distribution for {d.column}."

        # Determine directive based on median and 75th percentile gaps
        median_gap = next((g for p, t, c, g in gaps_by_percentile if p == "p50"), None)
        median_val = next((t for p, t, c, g in gaps_by_percentile if p == "p50"), None)
        p75_gap = next((g for p, t, c, g in gaps_by_percentile if p == "p75"), None)
        p75_val = next((t for p, t, c, g in gaps_by_percentile if p == "p75"), None)
        p25_gap = next((g for p, t, c, g in gaps_by_percentile if p == "p25"), None)
        p25_val = next((t for p, t, c, g in gaps_by_percentile if p == "p25"), None)
        p10_val = next((t for p, t, c, g in gaps_by_percentile if p == "p10"), None)
        p90_val = next((t for p, t, c, g in gaps_by_percentile if p == "p90"), None)

        parts: list[str] = []

        # Use explicit (a, b] notation: values > a and ≤ b
        # If p25 gap negative (target < current) → generate more in lower range
        if p25_gap and p25_gap < 0 and p10_val is not None and p25_val is not None:
            low_count = max(1, int(n_rows * 0.3))
            parts.append(f"{low_count} records in RANGE ({p10_val:.2f}, {p25_val:.2f}] (values > {p10_val:.2f} and ≤ {p25_val:.2f})")

        # If median gap negative (target < current) → generate more in lower-mid range
        if median_gap and median_gap < 0 and p25_val is not None and median_val is not None:
            low_count = max(1, int(n_rows * 0.3))
            parts.append(f"{low_count} records in RANGE ({p25_val:.2f}, {median_val:.2f}] (values > {p25_val:.2f} and ≤ {median_val:.2f})")

        # If p75 gap negative (target < current) → generate more in mid range
        if p75_gap and p75_gap < 0 and median_val is not None and p75_val is not None:
            mid_count = max(1, int(n_rows * 0.3))
            parts.append(f"{mid_count} records in RANGE ({median_val:.2f}, {p75_val:.2f}] (values > {median_val:.2f} and ≤ {p75_val:.2f})")

        # If median gap positive (target > current) → generate more in upper-mid range
        if median_gap and median_gap > 0 and median_val is not None and p75_val is not None:
            high_count = max(1, int(n_rows * 0.3))
            parts.append(f"{high_count} records in RANGE ({median_val:.2f}, {p75_val:.2f}] (values > {median_val:.2f} and ≤ {p75_val:.2f})")

        # If p75 gap positive (target > current) → generate more in upper range
        if p75_gap and p75_gap > 0 and p75_val is not None and p90_val is not None:
            high_count = max(1, int(n_rows * 0.3))
            parts.append(f"{high_count} records in RANGE ({p75_val:.2f}, {p90_val:.2f}] (values > {p75_val:.2f} and ≤ {p90_val:.2f})")

        if parts:
            return f"Generate {', '.join(parts)}."

        # Fallback
        return f"Adjust distribution shape for {d.column}."

    if d.gap_type == "sentinel_gap":
        target = float(d.target_value) if d.target_value is not None else 0.0
        current = float(d.current_value) if d.current_value is not None else None
        sentinel_val = d.metric.replace("sentinel=", "")
        count = _proportion_count_text(
            target, current, n_rows, current_row_count, target_total_rows
        )
        return f"Generate {count} where {d.column} = {sentinel_val}."

    if d.gap_type == "categorical_gap":
        target = float(d.target_value) if d.target_value is not None else 0.0
        current = float(d.current_value) if d.current_value is not None else None
        cat_name = d.metric.replace(" frequency", "").strip("'")
        count = _proportion_count_text(
            target, current, n_rows, current_row_count, target_total_rows
        )
        return f"Generate {count} where {d.column} = '{cat_name}'."

    if d.gap_type == "corr_gap":
        col_a, col_b = d.column if isinstance(d.column, list) else (d.column, d.column)
        target_r = float(d.target_value) if d.target_value is not None else 0.0
        current_r = float(d.current_value) if d.current_value is not None else 0.0
        gap = target_r - current_r  # gap = target - current

        if abs(target_r) < 0.15:
            # Near-zero target correlation
            if abs(gap) > 0.15:
                return f"Generate records where {col_a} and {col_b} vary independently (mix high {col_a} with both high and low {col_b})."
            return f"Mix {col_a} and {col_b} values independently across rows."

        if gap < 0:
            # Target correlation weaker than current → add scatter
            scatter_count = max(1, int(n_rows * 0.3))
            if target_r > 0:
                return f"Weaken positive correlation: generate {scatter_count} records with high {col_a} + low {col_b}, and {scatter_count} with low {col_a} + high {col_b}."
            else:
                return f"Weaken negative correlation: generate {scatter_count} records with high {col_a} + high {col_b}, and {scatter_count} with low {col_a} + low {col_b}."
        else:
            # Target correlation stronger than current → strengthen pattern
            strengthen_count = max(1, int(n_rows * 0.7))
            if target_r > 0:
                return f"Strengthen positive correlation: generate {strengthen_count} records with both high or both low ({col_a} + {col_b} in same range)."
            else:
                return f"Strengthen negative correlation: generate {strengthen_count} records with opposite ranges (high {col_a} + low {col_b} OR low {col_a} + high {col_b})."

    if d.gap_type == "contingency_gap":
        col_a, col_b = d.column if isinstance(d.column, list) else (d.column, d.column)
        if d.cells:
            parts: list[str] = []
            for cell in d.cells:
                a_val = cell.get("c1_val")
                b_val = cell.get("c2_val")
                freq = cell.get("freq")
                if a_val is None or b_val is None or freq is None:
                    continue
                count = _proportion_count_text(
                    float(freq),
                    current_proportion=None,
                    n_rows=n_rows,
                    current_row_count=0,
                    target_total_rows=target_total_rows,
                )
                parts.append(f"{count} with {col_a}={a_val} AND {col_b}={b_val}")
            if parts:
                return "Generate " + "; ".join(parts) + "."
        return f"Generate records matching the most common {col_a} × {col_b} pairings from real data."

    return d.note


def enrich_directives_with_outcome(
    directive_dicts: list[dict],
    after_df: pd.DataFrame,
    target_profiles: dict[str, NumericTextureProfile],
    after_report: dict[str, object] | None = None,
) -> list[dict]:
    """Add ``value_after``, ``improvement_pct``, and ``score_after`` to each gap directive dict.

    Computes post-generation metric values directly from *after_df* and
    compares them against the pre-generation ``current_value`` to measure
    how much each gap closed (positive = improved, negative = regressed).

    When *after_report* (an ``evaluate_datasets`` result comparing target to
    *after_df*) is provided, each directive also receives ``score_after`` — the
    SDMetrics column/pair quality score after this iteration.
    """
    after_column_shapes: dict[str, object] = {}
    after_column_pair_trends: dict[str, object] = {}
    if after_report is not None:
        after_column_shapes, after_column_pair_trends = _build_score_lookups(after_report)
    quantile_level_map = {
        "p1": 0.01,
        "p5": 0.05,
        "p10": 0.10,
        "p25": 0.25,
        "p50": 0.50,
        "p75": 0.75,
        "p90": 0.90,
        "p95": 0.95,
        "p99": 0.99,
    }
    result: list[dict] = []
    for d in directive_dicts:
        d = dict(d)
        gap_type = d.get("gap_type")
        target_value = d.get("target_value")
        current_value = d.get("current_value")
        value_after: float | None = None

        if gap_type == "sentinel_gap":
            col = str(d.get("column", ""))
            metric = d.get("metric", "")
            sentinel_val_str = metric.replace("sentinel=", "")
            try:
                sentinel_val = float(sentinel_val_str)
            except (ValueError, TypeError):
                sentinel_val = None
            if col in after_df.columns and sentinel_val is not None:
                series = pd.to_numeric(after_df[col], errors="coerce").dropna()
                if len(series) > 0:
                    value_after = round(float((series == sentinel_val).mean()), 4)

        elif gap_type == "quantile_gap":
            col = str(d.get("column", ""))
            metric = d.get("metric", "")
            cells = d.get("cells")
            # Collapsed debug-style directive: enrich each cell in place.
            if cells and col in after_df.columns:
                series = pd.to_numeric(after_df[col], errors="coerce").dropna()
                tp = target_profiles.get(col)
                if tp and tp.sentinels:
                    for s in tp.sentinels:
                        series = series[series != s.value]
                enriched_cells: list[dict] = []
                for cell in cells:
                    cell = dict(cell)
                    cell_metric = cell.get("metric", "")
                    cell_level = quantile_level_map.get(cell_metric)
                    if cell_level is not None and len(series) > 0:
                        cell["value_after"] = round(float(series.quantile(cell_level)), 2)
                    else:
                        cell["value_after"] = None
                    enriched_cells.append(cell)
                d["cells"] = enriched_cells
            else:
                # Legacy per-percentile directive path (still used by the main
                # prompt pipeline): enrich the single value_after from metric.
                level = quantile_level_map.get(metric)
                if col in after_df.columns and level is not None:
                    series = pd.to_numeric(after_df[col], errors="coerce").dropna()
                    tp = target_profiles.get(col)
                    if tp and tp.sentinels:
                        for s in tp.sentinels:
                            series = series[series != s.value]
                    if len(series) > 0:
                        value_after = round(float(series.quantile(level)), 2)

        elif gap_type == "categorical_gap":
            col = str(d.get("column", ""))
            metric = d.get("metric", "")
            try:
                cat = metric.split("'")[1]
            except IndexError:
                cat = None
            if col in after_df.columns and cat is not None:
                # Match evaluate_datasets' categorical coercion so value_after
                # buckets agree with score_after (both stringify + sentinel-fill).
                from llm_segment_repair.metrics import (
                    CATEGORICAL_NULL_SENTINEL,
                    _normalize_categorical_value,
                )

                normalized = after_df[col].map(
                    lambda v: _normalize_categorical_value(v, CATEGORICAL_NULL_SENTINEL)
                )
                counts = normalized.value_counts(normalize=True)
                value_after = round(float(counts.get(cat, 0.0)), 4)

        elif gap_type == "corr_gap":
            columns = d.get("column")
            if isinstance(columns, list) and len(columns) == 2:
                col_a, col_b = columns
                if col_a in after_df.columns and col_b in after_df.columns:
                    a = pd.to_numeric(after_df[col_a], errors="coerce")
                    b = pd.to_numeric(after_df[col_b], errors="coerce")
                    mask = a.notna() & b.notna()
                    if mask.sum() > 2:
                        value_after = round(float(a[mask].corr(b[mask])), 4)

        elif gap_type == "contingency_gap":
            # ContingencySimilarity is a real-vs-synthetic metric; it can only be
            # computed from after_report (which pairs target with after_df).
            columns = d.get("column")
            if isinstance(columns, list) and len(columns) == 2 and after_report is not None:
                col_a, col_b = str(columns[0]), str(columns[1])
                for entry in after_report.get("contingency_summary", []) or []:
                    e_c1 = str(entry.get("c1") or "")
                    e_c2 = str(entry.get("c2") or "")
                    score = entry.get("score")
                    if score is None:
                        continue
                    if {e_c1, e_c2} == {col_a, col_b}:
                        value_after = round(float(score), 4)
                        break

        d["value_after"] = value_after

        improvement_pct: float | None = None
        if value_after is not None and target_value is not None and current_value is not None:
            try:
                tv = float(target_value)
                cv = float(current_value)
                av = float(value_after)
                gap_before = abs(tv - cv)
                gap_after = abs(tv - av)
                if gap_before > 1e-9:
                    improvement_pct = round((gap_before - gap_after) / gap_before * 100, 1)
                elif gap_after < 1e-9:
                    improvement_pct = 100.0
                else:
                    improvement_pct = -100.0
            except (ValueError, TypeError):
                pass
        d["improvement_pct"] = improvement_pct

        if after_report is not None:
            d["score_after"] = _lookup_sdmetrics_score(
                d, after_column_shapes, after_column_pair_trends
            )

        result.append(d)
    return result


def compute_gap(
    real_df: pd.DataFrame, synth_df: pd.DataFrame, schema: ResolvedSchema
) -> dict[str, object]:
    from llm_segment_repair.metrics import evaluate_datasets

    report = evaluate_datasets(real_df, synth_df, schema=schema)
    missing_stats = _extract_missing_gap_statistics(report)

    gap: dict[str, dict[str, object] | list[str]] = {
        "categorical": {},
        "numeric": {},
        "correlation": {},
        "missing_statistics": missing_stats,
    }
    for entry in report.get("categorical_summary", []):
        col = entry.get("column")
        target_top = entry.get("top_categories_real", {}) or {}
        current_top = entry.get("top_categories_synthetic", {}) or {}
        deltas = {
            str(cat): round(
                abs(float(target_top.get(cat, 0.0)) - float(current_top.get(cat, 0.0))), 4
            )
            for cat in (set(target_top) | set(current_top))
        }
        top = sorted(deltas.items(), key=lambda x: x[1], reverse=True)[:3]
        gap["categorical"][str(col)] = top

    for entry in report.get("numerical_summary", []):
        col = str(entry.get("column"))
        real_mean = entry.get("real_mean")
        synthetic_mean = entry.get("synthetic_mean")
        real_median = entry.get("real_median")
        synthetic_median = entry.get("synthetic_median")
        if None in (real_mean, synthetic_mean, real_median, synthetic_median):
            continue
        gap["numeric"][col] = {
            "mean_delta": round(float(synthetic_mean) - float(real_mean), 3),
            "median_delta": round(float(synthetic_median) - float(real_median), 3),
        }

    corr = report.get("correlation", {})
    abs_diff = corr.get("abs_diff", {}) if isinstance(corr, dict) else {}
    corr_gap = []
    for i, col_a in enumerate(schema.correlation):
        row = abs_diff.get(col_a, {})
        for col_b in schema.correlation[i + 1 :]:
            val = row.get(col_b)
            if val is None:
                continue
            corr_gap.append((col_a, col_b, round(float(val), 3)))
    gap["correlation"] = sorted(corr_gap, key=lambda x: abs(x[2]), reverse=True)[:5]
    return gap


# =============================================================================
# ITERATIVE GAPS & CONVERGENCE
# =============================================================================


def compute_ranked_gaps(
    target_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    dataset_name: str,
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
    top_n: int,
) -> list[GapMetric]:
    """
    Compute and rank gaps between synthetic and target statistics.

    This function identifies the most significant discrepancies between the
    current synthetic data and the target distribution, returning them
    ranked by priority score.

    Args:
        target_df: DataFrame with real/target data.
        synthetic_df: DataFrame with current synthetic data.
        dataset_name: Name of the dataset for loading metadata.
        cat_cols: List of categorical column names.
        num_cols: List of numeric column names.
        top_n: Number of top gaps to return.

    Returns:
        List of GapMetric objects, sorted by priority_score descending.
    """
    from llm_segment_repair.metrics import evaluate_datasets

    report = evaluate_datasets(target_df, synthetic_df, schema=schema)
    missing_stats = _extract_missing_gap_statistics(report)
    if missing_stats:
        logger.warning(
            "missing_gap_statistics kind=ranked_gaps missing=%s",
            ",".join(missing_stats),
        )

    gaps: list[GapMetric] = []

    # Categorical frequency gaps
    for entry in report.get("categorical_summary", []):
        col = str(entry.get("column"))
        target_dist = entry.get("top_categories_real", {}) or {}
        synth_dist = entry.get("top_categories_synthetic", {}) or {}
        for category in set(target_dist) | set(synth_dist):
            target_freq = float(target_dist.get(category, 0.0))
            synth_freq = float(synth_dist.get(category, 0.0))
            gap = target_freq - synth_freq

            if abs(gap) > 0.01:  # Only report gaps > 1%
                direction = "increase" if gap > 0 else "decrease"
                gaps.append(
                    GapMetric(
                        attribute=str(col),
                        metric_type="categorical_freq",
                        target_value=round(target_freq, 4),
                        current_value=round(synth_freq, 4),
                        gap=round(gap, 4),
                        direction=direction,
                        priority_score=abs(gap),
                        detail=f"'{category}' frequency",
                    )
                )

    # Numeric mean gaps
    for entry in report.get("numerical_summary", []):
        col = str(entry.get("column"))
        target_mean = entry.get("real_mean")
        synth_mean = entry.get("synthetic_mean")
        if target_mean is not None and synth_mean is not None and float(target_mean) != 0:
            relative_gap = (float(target_mean) - float(synth_mean)) / abs(float(target_mean))
            if abs(relative_gap) > 0.02:  # Only report gaps > 2%
                direction = "increase" if relative_gap > 0 else "decrease"
                gaps.append(
                    GapMetric(
                        attribute=col,
                        metric_type="numeric_mean",
                        target_value=round(float(target_mean), 2),
                        current_value=round(float(synth_mean), 2),
                        gap=round(float(target_mean) - float(synth_mean), 2),
                        direction=direction,
                        priority_score=abs(relative_gap),
                        detail="mean",
                    )
                )

        # Also check median gaps
        target_median = entry.get("real_median")
        synth_median = entry.get("synthetic_median")
        if target_median is not None and synth_median is not None and float(target_median) != 0:
            relative_gap = (float(target_median) - float(synth_median)) / abs(float(target_median))
            if abs(relative_gap) > 0.02:
                direction = "increase" if relative_gap > 0 else "decrease"
                gaps.append(
                    GapMetric(
                        attribute=col,
                        metric_type="numeric_median",
                        target_value=round(float(target_median), 2),
                        current_value=round(float(synth_median), 2),
                        gap=round(float(target_median) - float(synth_median), 2),
                        direction=direction,
                        priority_score=abs(relative_gap) * 0.8,  # Slightly lower priority than mean
                        detail="median",
                    )
                )

    # Sort by priority score and return top N
    gaps.sort(key=lambda x: x.priority_score, reverse=True)
    return gaps[:top_n]


def format_priority_gaps(gaps: list[GapMetric]) -> str:
    """
    Format priority gaps as human-readable text for the LLM prompt.

    Args:
        gaps: List of GapMetric objects to format.

    Returns:
        Formatted string describing the gaps and correction directions.
    """
    if not gaps:
        return "No significant gaps detected. Continue generating diverse, realistic rows."

    lines = []
    for i, gap in enumerate(gaps, 1):
        if gap.metric_type == "categorical_freq":
            lines.append(
                f"{i}. {gap.attribute} ({gap.detail}): "
                f"target={gap.target_value:.1%}, current={gap.current_value:.1%} → "
                f"**{gap.direction.upper()}** this category by {abs(gap.gap):.1%}"
            )
        else:
            lines.append(
                f"{i}. {gap.attribute} ({gap.detail}): "
                f"target={gap.target_value:.1f}, current={gap.current_value:.1f} → "
                f"**{gap.direction.upper()}** values (gap={gap.gap:+.1f})"
            )
    return "\n".join(lines)


def check_convergence(
    target_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    schema: ResolvedSchema,
    threshold: float,
    corr_weight: float = 0.3,
    report: dict[str, object] | None = None,
) -> tuple[bool, float]:
    """
    Check if synthetic data has converged to target distribution.

    Convergence is a weighted blend of:
    - Mean absolute relative error across numeric column means.
    - Mean absolute correlation error across numeric column pairs
      (when *corr_cols* is provided).

    Args:
        target_df: DataFrame with real/target data.
        synthetic_df: DataFrame with current synthetic data.
        dataset_name: Name of the dataset for loading metadata.
        num_cols: List of numeric column names.
        threshold: Convergence threshold (e.g., 0.05 = 5% error).
        corr_cols: Columns to use for correlation error. If *None*,
            correlation error is not included (backwards-compatible).
        corr_weight: Weight for correlation error in the blended metric
            (0.0–1.0).  Mean-error weight is ``1 - corr_weight``.

    Returns:
        Tuple of (converged: bool, blended_error: float).
    """
    assert report is not None
    missing_stats = _extract_missing_gap_statistics(report)
    if missing_stats:
        logger.warning(
            "missing_gap_statistics kind=convergence missing=%s",
            ",".join(missing_stats),
        )

    # --- mean error -------------------------------------------------------
    errors: list[float] = []
    for entry in report.get("numerical_summary", []):
        real_mean = entry.get("real_mean")
        synthetic_mean = entry.get("synthetic_mean")
        if real_mean is None or synthetic_mean is None:
            continue
        real_mean_float = float(real_mean)
        synthetic_mean_float = float(synthetic_mean)
        if real_mean_float == 0:
            continue
        relative_error = float(abs(real_mean_float - synthetic_mean_float) / abs(real_mean_float))
        errors.append(relative_error)

    mean_error = float(sum(errors) / len(errors)) if errors else 0.0

    # --- correlation error ------------------------------------------------
    corr_error = 0.0
    if schema.correlation and len(schema.correlation) >= 2:
        corr_report = report
        abs_diff = (corr_report.get("correlation") or {}).get("abs_diff", {})
        corr_diffs: list[float] = []
        for idx, ca in enumerate(schema.numerical):
            row = abs_diff.get(ca, {})
            for cb in schema.numerical[idx + 1 :]:
                val = row.get(cb)
                if val is not None and np.isfinite(val):
                    corr_diffs.append(float(val))
        if corr_diffs:
            corr_error = float(sum(corr_diffs) / len(corr_diffs))

    # --- blend ------------------------------------------------------------
    if schema.categorical and len(schema.numerical) >= 2:
        blended = (1 - corr_weight) * mean_error + corr_weight * corr_error
    else:
        blended = mean_error

    converged = bool(blended <= threshold)
    return converged, blended


def evaluate_iteration_acceptance(
    previous_overall: float | None,
    candidate_overall: float | None,
    previous_frobenius: float | None,
    candidate_frobenius: float | None,
    *,
    min_overall_score_delta: float,
    max_frobenius_increase: float,
) -> tuple[bool, list[str], float | None, float | None]:
    """Return whether a candidate iteration should be accepted under objective guardrails."""
    reasons: list[str] = []

    overall_delta = None
    if previous_overall is not None and candidate_overall is not None:
        overall_delta = candidate_overall - previous_overall
        if overall_delta < min_overall_score_delta:
            reasons.append(
                f"overall_delta_below_threshold:{overall_delta:.6f}<{min_overall_score_delta:.6f}"
            )
    elif previous_overall is not None and candidate_overall is None:
        reasons.append("candidate_overall_unavailable")

    # Frobenius is a distance-to-target gap; we guard against increases in that gap.
    frobenius_delta = None
    if previous_frobenius is not None and candidate_frobenius is not None:
        frobenius_delta = candidate_frobenius - previous_frobenius
        if frobenius_delta > max_frobenius_increase:
            reasons.append(
                "frobenius_gap_increase_above_threshold:"
                f"{frobenius_delta:.6f}>{max_frobenius_increase:.6f}"
            )
    elif previous_frobenius is not None and candidate_frobenius is None:
        reasons.append("candidate_frobenius_unavailable")

    return len(reasons) == 0, reasons, overall_delta, frobenius_delta
