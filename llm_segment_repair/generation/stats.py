"""Statistical summaries and formatting helpers for prompts."""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .core import NumericTextureProfile, SentinelInfo
from .schema import ResolvedSchema


def _format_distribution_pct(dist: dict[str, float]) -> str:
    """Render a categorical distribution as a JSON object of percentage strings."""
    return json.dumps({k: f"{round(float(v) * 100, 1)}%" for k, v in dist.items()})


def summarise_categorical(
    df: pd.DataFrame, columns: Sequence[str], top_k: int
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        counts = df[col].astype(str).value_counts(dropna=False, normalize=True)
        summary[col] = {str(idx): round(float(prop), 4) for idx, prop in counts.head(top_k).items()}
    return summary


def summarise_numeric(df: pd.DataFrame, columns: Sequence[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        stats[col] = {
            "mean": round(float(series.mean()), 3),
            "median": round(float(series.median()), 3),
            "p25": round(float(series.quantile(0.25)), 3),
            "p75": round(float(series.quantile(0.75)), 3),
            "min": round(float(series.min()), 3),
            "max": round(float(series.max()), 3),
        }
    return stats


def summarise_numeric_extended(
    df: pd.DataFrame, columns: Sequence[str]
) -> dict[str, dict[str, object]]:
    """
    Compute extended numeric statistics for synthetic data generation.

    This function provides detailed statistics to help the LLM understand
    the distribution characteristics of numeric columns, including:
    - Support (min/max range)
    - Mass (proportion of zeros)
    - Full quantile profile
    - Heaping signature (rounding patterns)
    - Log-transformed statistics

    Args:
        df: DataFrame containing the data.
        columns: List of numeric column names to summarize.

    Returns:
        Dictionary mapping column names to their extended statistics.
    """
    stats: dict[str, dict[str, object]] = {}

    for col in columns:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(series) == 0:
            continue

        col_stats: dict[str, object] = {}

        # Support: min and max
        col_stats["support"] = {
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
        }

        # Mass: proportion of zeros
        n_total = len(series)
        n_zeros = (series == 0).sum()
        col_stats["mass"] = {
            "p_zero": round(float(n_zeros / n_total), 4) if n_total > 0 else 0.0,
        }

        # Quantiles: full profile
        quantile_levels = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
        quantile_names = ["p0", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p100"]
        col_stats["quantiles"] = {
            name: round(float(series.quantile(level)), 2)
            for name, level in zip(quantile_names, quantile_levels)
        }

        # Heaping signature: rounding patterns in the data
        # This helps the LLM understand if values tend to be integers or rounded
        non_zero_series = series[series != 0]
        if len(non_zero_series) > 0:
            # Percentage that are integers (no decimal part)
            pct_int = (non_zero_series == non_zero_series.astype(int)).mean()
            # Percentage divisible by 5
            pct_mod_5 = (non_zero_series % 5 == 0).mean()
            # Percentage divisible by 10
            pct_mod_10 = (non_zero_series % 10 == 0).mean()
        else:
            pct_int = 0.0
            pct_mod_5 = 0.0
            pct_mod_10 = 0.0

        col_stats["heaping"] = {
            "pct_int": round(float(pct_int), 4),
            "pct_mod_5": round(float(pct_mod_5), 4),
            "pct_mod_10": round(float(pct_mod_10), 4),
        }

        # Log1p statistics: useful for right-skewed expenditure data
        # log1p(x) = log(1 + x), handles zeros gracefully
        log_series = np.log1p(series)
        col_stats["log1p_stats"] = {
            "mu_log": round(float(log_series.mean()), 4),
            "sigma_log": round(float(log_series.std()), 4),
        }

        stats[col] = col_stats

    return stats


def format_numeric_extended(stats: dict[str, dict[str, object]]) -> str:
    """
    Format extended numeric statistics as structured text for the LLM prompt.

    Args:
        stats: Extended statistics from summarise_numeric_extended().

    Returns:
        Formatted multi-line string describing each column's statistics.
    """
    if not stats:
        return "No numeric statistics available."

    lines = []
    for col, col_stats in stats.items():
        lines.append(f"\n**{col}:**")

        # Support
        support = col_stats.get("support", {})
        lines.append(f"  support: min={support.get('min', 'N/A')}, max={support.get('max', 'N/A')}")

        # Mass
        mass = col_stats.get("mass", {})
        lines.append(f"  mass: p_zero={mass.get('p_zero', 'N/A')}")

        # Quantiles
        quantiles = col_stats.get("quantiles", {})
        q_str = ", ".join(f"{k}={v}" for k, v in quantiles.items())
        lines.append(f"  quantiles: {q_str}")

        # Heaping
        heaping = col_stats.get("heaping", {})
        lines.append(
            f"  heaping: pct_int={heaping.get('pct_int', 'N/A')}, "
            f"pct_mod_5={heaping.get('pct_mod_5', 'N/A')}, "
            f"pct_mod_10={heaping.get('pct_mod_10', 'N/A')}"
        )

        # Log1p stats
        log_stats = col_stats.get("log1p_stats", {})
        lines.append(
            f"  log1p_stats: mu_log={log_stats.get('mu_log', 'N/A')}, "
            f"sigma_log={log_stats.get('sigma_log', 'N/A')}"
        )

    return "\n".join(lines)


# =============================================================================
# DATASET-AGNOSTIC NUMERIC TEXTURE PROFILING
# =============================================================================


def compute_decimal_places_dist(series: pd.Series, max_decimals: int = 6) -> dict[int, float]:
    """
    Compute distribution of decimal places used in a numeric series.

    For each value, determines how many decimal places are actually used
    (after stripping trailing zeros).

    Args:
        series: Numeric pandas Series.
        max_decimals: Maximum decimals to consider.

    Returns:
        Dictionary mapping decimal count to frequency.
    """

    def count_decimals(v: float) -> int:
        if pd.isna(v) or v == 0:
            return 0
        # Convert to string, strip trailing zeros
        s = f"{v:.{max_decimals}f}".rstrip("0").rstrip(".")
        if "." not in s:
            return 0
        return len(s.split(".")[1])

    decimal_counts = series.apply(count_decimals)
    dist = decimal_counts.value_counts(normalize=True).to_dict()
    # Ensure sorted by decimal count
    return {k: round(v, 4) for k, v in sorted(dist.items())}


def compute_fractional_topk(
    series: pd.Series, precision: int = 2, top_k: int = 5
) -> list[tuple[str, float]]:
    """
    Compute most common fractional endings at a given precision.

    Args:
        series: Numeric pandas Series.
        precision: Decimal precision to round to (usually 2).
        top_k: Number of top endings to return.

    Returns:
        List of (ending_string, frequency) tuples, e.g., [(".00", 0.28), (".50", 0.07)]
    """
    non_zero = series[series != 0].dropna()
    if len(non_zero) == 0:
        return []

    def get_fractional(v: float) -> str:
        frac = round(v - int(v), precision)
        return f".{abs(frac):.{precision}f}".split(".")[1]

    fracs = non_zero.apply(get_fractional)
    counts = fracs.value_counts(normalize=True)
    return [(f".{k}", round(float(v), 4)) for k, v in counts.head(top_k).items()]


def compute_gridness_topk(
    series: pd.Series, candidates: list[int] | None = None, top_k: int = 3
) -> list[tuple[str, float]]:
    """
    Detect step-size signatures (gridness/heaping patterns) in numeric data.

    Instead of hardcoding specific divisors, this detects which step sizes
    are most common in the data.

    Args:
        series: Numeric pandas Series.
        candidates: Candidate divisors to check. Default: [2, 5, 10, 25, 50, 100]
        top_k: Number of top patterns to return.

    Returns:
        List of (pattern_name, frequency) tuples.
    """
    if candidates is None:
        candidates = [2, 5, 10, 25, 50, 100]

    non_zero = series[series != 0].dropna()
    if len(non_zero) == 0:
        return []

    patterns = []
    for divisor in candidates:
        # Check what fraction of non-zero values are divisible by this
        divisible = (non_zero % divisor == 0).mean()
        if divisible > 0.05:  # Only report if > 5%
            patterns.append((f"divisible_by_{divisor}", round(float(divisible), 4)))

    # Sort by frequency descending
    patterns.sort(key=lambda x: x[1], reverse=True)
    return patterns[:top_k]


def _detect_extreme_sentinels(
    series: pd.Series,
    quantile_keys: list[str],
    quantile_levels: dict[str, float],
) -> list[SentinelInfo]:
    """Detect point-mass sentinels at extremes of a numeric series.

    A sentinel is identified when multiple consecutive quantiles from
    the left (or right) tail share the exact same value AND that value
    equals the series min (or max).  The mass is estimated from how
    deep into the quantile profile the flat region extends.

    Args:
        series: Numeric pandas Series (NaN-free).
        quantile_keys: Ordered quantile names, e.g. ["p1","p5",..."p99"].
        quantile_levels: Mapping of quantile name to level, e.g. {"p1": 0.01, ...}.

    Returns:
        List of SentinelInfo (may be empty, or have one left and/or one right entry).
    """
    if len(series) == 0:
        return []

    raw_quantiles = {
        name: round(float(series.quantile(level)), 2)
        for name, level in quantile_levels.items()
    }

    sentinels: list[SentinelInfo] = []

    series_min = round(float(series.min()), 2)
    series_max = round(float(series.max()), 2)

    # --- left sentinel ---------------------------------------------------
    # Walk from p1 rightward; as long as the quantile equals the series min
    # the flat region continues.
    left_depth = 0.0
    for key in quantile_keys:
        if raw_quantiles[key] == series_min:
            left_depth = quantile_levels[key]
        else:
            break
    # Require at least two consecutive quantiles to be flat (>= p5 level)
    if left_depth >= 0.05:
        exact_mass = round(float((series == series_min).mean()), 4)
        sentinels.append(SentinelInfo(value=series_min, mass=exact_mass, side="left"))

    # --- right sentinel --------------------------------------------------
    right_depth = 0.0
    for key in reversed(quantile_keys):
        if raw_quantiles[key] == series_max:
            right_depth = 1.0 - quantile_levels[key]
        else:
            break
    if right_depth >= 0.05:
        exact_mass = round(float((series == series_max).mean()), 4)
        sentinels.append(SentinelInfo(value=series_max, mass=exact_mass, side="right"))

    return sentinels


def compute_numeric_texture_profile(
    df: pd.DataFrame,
    column: str,
    target_sentinels: list[SentinelInfo] | None = None,
    disable_sentinels: bool = False,
) -> NumericTextureProfile | None:
    """
    Compute a complete NumericTextureProfile for a single column.

    This is a universal, dataset-agnostic profile that captures both
    distribution shape and numeric texture.

    When sentinels (point-mass atoms at extremes) are detected, the
    quantiles and support are recomputed on the *continuous* portion
    after stripping the sentinel values.  The sentinel metadata is
    stored in ``sentinels`` so downstream formatting can describe the
    mixture explicitly.

    Args:
        df: DataFrame containing the data.
        column: Name of the numeric column to profile.
        target_sentinels: When profiling current synthetic data, pass
            the target's sentinel list so the same sentinel values are
            reused instead of detecting new (wrong) ones.

    Returns:
        NumericTextureProfile or None if column doesn't exist or is empty.
    """
    if column not in df.columns:
        return None

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(series) == 0:
        return None

    # Shape: p_zero (computed on full series, before sentinel removal)
    p_zero = round(float((series == 0).mean()), 4)

    # Quantile key / level mapping (reused for sentinel detection)
    quantile_keys = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
    quantile_levels = {
        "p1": 0.01, "p5": 0.05, "p10": 0.10, "p25": 0.25, "p50": 0.50,
        "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99,
    }

    if disable_sentinels:
        # Ablation mode: skip detection entirely, compute quantiles/support on
        # the raw series so the LLM sees the point-mass atoms as part of the
        # distribution rather than as separately-described SENTINEL lines.
        sentinels = []
    elif target_sentinels is not None:
        sentinels = [
            SentinelInfo(
                value=s.value,
                mass=round(float((series == s.value).mean()), 4),
                side=s.side,
            )
            for s in target_sentinels
        ]
    else:
        sentinels = _detect_extreme_sentinels(series, quantile_keys, quantile_levels)

    # Build a continuous sub-series with sentinels stripped out
    continuous = series
    for s in sentinels:
        continuous = continuous[continuous != s.value]

    if len(continuous) == 0:
        # Entire column is sentinel(s) — keep raw stats
        support = (round(float(series.min()), 4), round(float(series.max()), 4))
        quantiles = {
            name: round(float(series.quantile(level)), 2)
            for name, level in quantile_levels.items()
        }
        return NumericTextureProfile(
            column=column, min=support[0], max=support[1],
            p_zero=p_zero, quantiles=quantiles, sentinels=sentinels or None,
        )

    # Support and quantiles on the continuous portion
    support = (round(float(continuous.min()), 4), round(float(continuous.max()), 4))
    quantiles = {
        name: round(float(continuous.quantile(level)), 2)
        for name, level in quantile_levels.items()
    }

    return NumericTextureProfile(
        column=column,
        min=support[0],
        max=support[1],
        p_zero=p_zero,
        quantiles=quantiles,
        sentinels=sentinels or None,
    )


def compute_all_texture_profiles(
    df: pd.DataFrame,
    columns: Sequence[str],
    target_profiles: dict[str, NumericTextureProfile] | None = None,
    disable_sentinels: bool = False,
) -> dict[str, NumericTextureProfile]:
    """
    Compute NumericTextureProfiles for all specified columns.

    Args:
        df: DataFrame containing the data.
        columns: List of numeric column names.
        target_profiles: When profiling current synthetic data, pass
            the target profiles so sentinel values are reused from the
            target rather than independently detected.

    Returns:
        Dictionary mapping column names to their profiles.
    """
    profiles = {}
    for col in columns:
        target_sentinels = None
        if target_profiles and col in target_profiles:
            target_sentinels = target_profiles[col].sentinels
        profile = compute_numeric_texture_profile(
            df,
            col,
            target_sentinels=target_sentinels,
            disable_sentinels=disable_sentinels,
        )
        if profile is not None:
            profiles[col] = profile
    return profiles


def _format_sentinel_lines(sentinels: list[SentinelInfo]) -> list[str]:
    """Return human-readable lines describing sentinel atoms."""
    lines: list[str] = []
    for s in sentinels:
        pct = round(s.mass * 100, 1)
        lines.append(
            f"  SENTINEL ({s.side}): value={s.value}, mass={pct}% of rows. "
            f"Stripped from quantiles/support below; generate this value separately."
        )
    return lines


def format_texture_profile(profile: NumericTextureProfile) -> str:
    """
    Format a NumericTextureProfile as structured text for the LLM prompt.

    When sentinels are present the output first describes the sentinel
    atoms, then shows the continuous-portion stats.

    Args:
        profile: The texture profile to format.

    Returns:
        Formatted multi-line string.
    """
    lines = [f"\n**{profile.column}:**"]

    if profile.sentinels:
        lines.extend(_format_sentinel_lines(profile.sentinels))
        lines.append("  [continuous portion after sentinel removal]")

    lines.append(f"  min: {profile.min:.2f}, max: {profile.max:.2f}")
    lines.append(f"  p_zero: {profile.p_zero}")
    q_str = ", ".join(f"{k}={v}" for k, v in profile.quantiles.items())
    lines.append(f"  quantiles: {{{q_str}}}")

    return "\n".join(lines)


def format_all_texture_profiles(profiles: dict[str, NumericTextureProfile]) -> str:
    """
    Format all texture profiles as structured text.

    Args:
        profiles: Dictionary of column name to profile.

    Returns:
        Formatted multi-line string for all profiles.
    """
    if not profiles:
        return "No numeric texture profiles available."

    return "\n".join(format_texture_profile(p) for p in profiles.values())


def summarise_correlations(
    df: pd.DataFrame, columns: Sequence[str]
) -> list[tuple[str, str, float]]:
    """Return all pairwise Pearson correlations for *columns*, sorted by magnitude."""
    columns = [col for col in columns if col in df.columns]
    if len(columns) < 2:
        return []

    # Coerce to numeric so mixed/object columns do not disappear unexpectedly
    # from the correlation matrix and cause key errors on lookup.
    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")
    valid_cols = [col for col in columns if numeric_df[col].notna().sum() >= 2]
    if len(valid_cols) < 2:
        return []

    corr = numeric_df[valid_cols].corr()
    pairs: list[tuple[str, str, float]] = []
    for i, col_a in enumerate(valid_cols):
        for col_b in valid_cols[i + 1 :]:
            if col_a not in corr.index or col_b not in corr.columns:
                continue
            val = corr.loc[col_a, col_b]
            if pd.isna(val):
                continue
            pairs.append((col_a, col_b, round(float(val), 3)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


# Subsets of `NumericTextureProfile.quantiles` keys used for ablating how many
# percentiles the LLM sees in the schema block. The profile itself always
# computes all 9 anchors (p1..p99); this map only affects what's *rendered*
# into the prompt. `None` preserves the default (emit all).
PERCENTILE_SUBSETS: dict[int, list[str]] = {
    1: ["p50"],
    3: ["p25", "p50", "p75"],
    5: ["p10", "p25", "p50", "p75", "p90"],
    9: ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"],
}


def _select_quantiles(
    quantiles: dict[str, float], num_percentiles: int | None
) -> dict[str, float]:
    """Return the subset of *quantiles* matching `num_percentiles` (preserving profile order)."""
    if num_percentiles is None:
        return dict(quantiles)
    if num_percentiles not in PERCENTILE_SUBSETS:
        raise ValueError(
            f"num_percentiles={num_percentiles!r} not in PERCENTILE_SUBSETS "
            f"{sorted(PERCENTILE_SUBSETS)}"
        )
    allowed = set(PERCENTILE_SUBSETS[num_percentiles])
    return {k: v for k, v in quantiles.items() if k in allowed}


def generate_schema_block(
    schema: ResolvedSchema,
    categorical_summary: dict[str, dict[str, float]],
    texture_profiles: dict[str, NumericTextureProfile | dict[str, float]] | None = None,
    null_rates: dict[str, float] | None = None,
    current_categorical_summary: dict[str, dict[str, float]] | None = None,
    current_texture_profiles: dict[str, NumericTextureProfile] | None = None,
    num_percentiles: int | None = None,
) -> str:
    """Generate a structured schema block describing each column.

    For numeric columns the block embeds the full
    ``NumericTextureProfile`` (min, max, p_zero, quantiles) so that the
    LLM has all distribution details in one place.  When current profiles
    are provided, both target and current values are shown side-by-side.
    """
    schema_lines = []

    for col in schema.columns:
        schema_lines.append(f"\nCOLUMN NAME: {col}")

        if null_rates and col in null_rates:
            pct = round(null_rates[col] * 100, 1)
            if pct > 0:
                schema_lines.append(f"NULL_RATE: {pct}%")

        # Boolean fields (even though in cat_cols)
        if col in schema.boolean:
            schema_lines.append("TYPE: boolean")
            schema_lines.append("VALUES: True or False")
            if col in categorical_summary:
                schema_lines.append(
                    f"TARGET DISTRIBUTION: {_format_distribution_pct(categorical_summary[col])}"
                )

        elif col in schema.categorical:
            if col in categorical_summary:
                categories = list(categorical_summary[col].keys())
            else:
                categories = []
            if len(categories) == 1:
                schema_lines.append(f"TYPE: categorical (fixed value: {json.dumps(categories[0])})")
            else:
                schema_lines.append("TYPE: categorical")
                if col in categorical_summary:
                    schema_lines.append(
                        f"TARGET MARGINAL DISTRIBUTION: {_format_distribution_pct(categorical_summary[col])}"
                    )

        elif col in schema.numerical:
            schema_lines.append("TYPE: float")

            cur_profile = (
                current_texture_profiles.get(col)
                if current_texture_profiles
                else None
            )

            if texture_profiles and col in texture_profiles:
                profile = texture_profiles[col]
                if isinstance(profile, dict):
                    min_v = profile.get("min")
                    max_v = profile.get("max")
                    if min_v is not None and max_v is not None:
                        schema_lines.append(f"  target_min: {float(min_v):.2f}, target_max: {float(max_v):.2f}")
                    if cur_profile:
                        schema_lines.append(f"  current_min: {cur_profile.min:.2f}, current_max: {cur_profile.max:.2f}")
                    else:
                        schema_lines.append("  current_min: n/a, current_max: n/a")
                    quantiles_map = {
                        "p25": profile.get("p25"),
                        "p50": profile.get("median"),
                        "p75": profile.get("p75"),
                    }
                    quantiles_map = {k: v for k, v in quantiles_map.items() if v is not None}
                    quantiles_map = _select_quantiles(quantiles_map, num_percentiles)
                    if quantiles_map:
                        q_str = ", ".join(f"{k}={v}" for k, v in quantiles_map.items())
                        schema_lines.append(f"  target_quantiles: {{{q_str}}}")
                    if cur_profile:
                        cur_q_map = _select_quantiles(cur_profile.quantiles, num_percentiles)
                        cur_q = ", ".join(f"{k}={v}" for k, v in cur_q_map.items())
                        schema_lines.append(f"  current_quantiles: {{{cur_q}}}")
                    else:
                        schema_lines.append("  current_quantiles: n/a")
                else:
                    _append_numeric_profile_lines(
                        schema_lines, profile, cur_profile, num_percentiles=num_percentiles
                    )

        else:
            schema_lines.append("TYPE: text")

    return "\n".join(schema_lines)


def _append_numeric_profile_lines(
    lines: list[str],
    target: NumericTextureProfile,
    current: NumericTextureProfile | None,
    num_percentiles: int | None = None,
) -> None:
    """Append target/current numeric profile lines to *lines*."""
    # Sentinels
    if target.sentinels:
        lines.extend(
            f"  TARGET {l}" for l in _format_sentinel_lines(target.sentinels)
        )
    if current and current.sentinels:
        lines.extend(
            f"  CURRENT {l}" for l in _format_sentinel_lines(current.sentinels)
        )
    elif target.sentinels:
        lines.append("  CURRENT SENTINEL: n/a")

    if target.sentinels:
        lines.append("  [continuous portion after sentinel removal]")

    # Min / max
    lines.append(f"  target_min: {target.min:.2f}, target_max: {target.max:.2f}")
    if current:
        lines.append(f"  current_min: {current.min:.2f}, current_max: {current.max:.2f}")
    else:
        lines.append("  current_min: n/a, current_max: n/a")

    # p_zero
    lines.append(f"  target_p_zero: {target.p_zero}")
    if current:
        lines.append(f"  current_p_zero: {current.p_zero}")
    else:
        lines.append("  current_p_zero: n/a")

    # Quantiles
    target_q = _select_quantiles(target.quantiles, num_percentiles)
    tq = ", ".join(f"{k}={v}" for k, v in target_q.items())
    lines.append(f"  target_quantiles: {{{tq}}}")
    if current:
        current_q = _select_quantiles(current.quantiles, num_percentiles)
        cq = ", ".join(f"{k}={v}" for k, v in current_q.items())
        lines.append(f"  current_quantiles: {{{cq}}}")
    else:
        lines.append("  current_quantiles: n/a")


def _interpret_correlation(r: float) -> str:
    """Return a plain-language interpretation of a Pearson r value."""
    ar = abs(r)
    if ar < 0.1:
        return "negligible — treat as independent"
    if ar < 0.3:
        return "weak — mostly independent, only a slight tendency"
    if ar < 0.5:
        return "moderate — noticeable trend but LOTS of scatter/exceptions"
    if ar < 0.7:
        return "moderate-strong — clear trend but still significant noise"
    return "strong — consistent co-movement with few exceptions"


def _scatter_recipe(r: float) -> tuple[int, int, int]:
    """Return (same_tier%, adjacent_tier%, independent%) for a target correlation.

    These fractions tell the LLM how often to place two correlated columns
    in the same quantile tier vs neighbouring vs fully independent tiers.
    """
    ar = abs(r)
    if ar < 0.1:
        return (33, 33, 34)  # effectively independent
    if ar < 0.3:
        return (30, 35, 35)
    if ar < 0.5:
        return (35, 35, 30)
    if ar < 0.7:
        return (45, 35, 20)
    return (55, 30, 15)


def _scatter_recipe_text(r: float) -> str:
    """Format the scatter recipe as a compact instruction string."""
    same, adj, indep = _scatter_recipe(r)
    return f"SAME-TIER ~{same}%, ADJACENT-TIER ~{adj}%, INDEPENDENT ~{indep}%"


TIER_LABELS = ["LOW", "MID", "HIGH"]

_NORMAL_TERCILE = 0.4307  # boundary for equal terciles of N(0,1)


def _compute_tier_boundaries(
    tp: NumericTextureProfile,
) -> tuple[float, float]:
    """Return (low_high_boundary, mid_high_boundary) from support range thirds."""
    support_range = tp.max - tp.min
    low_upper = tp.min + support_range / 3
    mid_upper = tp.min + 2 * support_range / 3
    return low_upper, mid_upper


def _count_current_tiers(
    current_df: pd.DataFrame | None,
    col: str,
    tp: NumericTextureProfile,
) -> dict[str, int]:
    """Count how many current rows fall in each tier (excluding sentinels)."""
    counts = {"LOW": 0, "MID": 0, "HIGH": 0}
    if current_df is None or col not in current_df.columns:
        return counts
    series = pd.to_numeric(current_df[col], errors="coerce").dropna()
    sentinel_values = {s.value for s in (tp.sentinels or [])}
    series = series[~series.isin(sentinel_values)]
    if len(series) == 0:
        return counts
    low_upper, mid_upper = _compute_tier_boundaries(tp)
    counts["LOW"] = int((series <= low_upper).sum())
    counts["MID"] = int(((series > low_upper) & (series <= mid_upper)).sum())
    counts["HIGH"] = int((series > mid_upper).sum())
    return counts


def compute_tier_blueprint(
    n_rows: int,
    columns: Sequence[str],
    target_corr_pairs: list[tuple[str, str, float]],
    target_profiles: dict[str, NumericTextureProfile] | None = None,
    current_profiles: dict[str, NumericTextureProfile] | None = None,
    current_row_count: int = 0,
    current_df: pd.DataFrame | None = None,
) -> list[dict[str, str]]:
    """Pre-compute tier assignments encoding correlation, sentinel mass, and exact tier counts.

    Computes exact batch tier counts to prevent overshoot, then uses a
    Gaussian copula to produce correlated samples and reassigns tiers via
    rank-ordering to match the target counts.
    """
    if len(columns) < 2 or n_rows == 0:
        return []

    col_idx = {c: i for i, c in enumerate(columns)}
    n = len(columns)

    # Build target correlation matrix from pairs
    corr_matrix = np.eye(n)
    for a, b, r in target_corr_pairs:
        if a in col_idx and b in col_idx:
            i, j = col_idx[a], col_idx[b]
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    # Ensure positive semi-definite via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    eigvals = np.maximum(eigvals, 1e-6)
    corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d_diag = np.sqrt(np.diag(corr_matrix))
    corr_matrix = corr_matrix / np.outer(d_diag, d_diag)

    # --- Compute exact batch tier counts per column ---
    total_after = current_row_count + n_rows
    batch_tier_targets: dict[str, dict[str, int]] = {}
    batch_sentinel_targets: dict[str, list[tuple[float, int]]] = {}

    for col in columns:
        tp = target_profiles.get(col) if target_profiles else None
        if tp is None:
            batch_tier_targets[col] = {
                "LOW": round(n_rows / 3),
                "MID": round(n_rows / 3),
                "HIGH": n_rows - 2 * round(n_rows / 3),
            }
            continue

        # Sentinel counts for this batch
        sentinel_batch_total = 0
        col_sentinels: list[tuple[float, int]] = []
        if tp.sentinels:
            for s in tp.sentinels:
                target_sent_count = s.mass * total_after
                cp = current_profiles.get(col) if current_profiles else None
                current_sent_mass = 0.0
                if cp and cp.sentinels:
                    for cs in cp.sentinels:
                        if cs.value == s.value:
                            current_sent_mass = cs.mass
                            break
                current_sent_count = current_sent_mass * current_row_count
                batch_need = max(0, round(target_sent_count - current_sent_count))
                batch_need = min(batch_need, n_rows)
                col_sentinels.append((s.value, batch_need))
                sentinel_batch_total += batch_need
        batch_sentinel_targets[col] = col_sentinels

        # Continuous tier counts (rows not assigned to sentinels)
        continuous_batch = n_rows - sentinel_batch_total
        if continuous_batch <= 0:
            batch_tier_targets[col] = {"LOW": 0, "MID": 0, "HIGH": 0}
            continue

        # Target: equal thirds of the continuous portion in the final dataset
        current_tiers = _count_current_tiers(current_df, col, tp)
        current_continuous = sum(current_tiers.values())
        target_continuous_final = total_after - round(
            sum(s.mass for s in (tp.sentinels or [])) * total_after
        )

        target_per_tier = target_continuous_final / 3
        batch_counts: dict[str, int] = {}
        for tier in ["LOW", "MID", "HIGH"]:
            need = max(0, round(target_per_tier - current_tiers[tier]))
            batch_counts[tier] = need

        # Normalise to fit continuous_batch exactly
        total_need = sum(batch_counts.values())
        if total_need > 0:
            for tier in batch_counts:
                batch_counts[tier] = round(batch_counts[tier] * continuous_batch / total_need)
            # Fix rounding to match exactly
            diff = continuous_batch - sum(batch_counts.values())
            if diff != 0:
                # Add/subtract from the tier with the largest need
                adjust_tier = max(batch_counts, key=lambda t: batch_counts[t])
                batch_counts[adjust_tier] += diff
        else:
            # No correction needed — distribute evenly
            batch_counts = {
                "LOW": round(continuous_batch / 3),
                "MID": round(continuous_batch / 3),
                "HIGH": continuous_batch - 2 * round(continuous_batch / 3),
            }

        batch_tier_targets[col] = batch_counts

    # --- Generate correlated samples via copula ---
    rng = np.random.default_rng()
    samples = rng.multivariate_normal(np.zeros(n), corr_matrix, size=n_rows)

    # --- Assign tiers via rank-ordering to match exact counts ---
    blueprint: list[dict[str, str]] = [{} for _ in range(n_rows)]

    for col in columns:
        col_i = col_idx[col]
        col_values = samples[:, col_i]
        tier_counts = batch_tier_targets[col]
        sentinels = batch_sentinel_targets.get(col, [])

        # Sort row indices by copula value (ascending)
        sorted_indices = list(np.argsort(col_values))

        # Assign sentinels first (to the lowest-ranked rows)
        sentinel_assigned = 0
        for s_val, s_count in sentinels:
            for idx in sorted_indices[:s_count]:
                blueprint[idx][col] = f"SENTINEL({s_val})"
            sorted_indices = sorted_indices[s_count:]
            sentinel_assigned += s_count

        # Assign LOW, MID, HIGH in order to remaining sorted rows
        pos = 0
        for tier in ["LOW", "MID", "HIGH"]:
            count = tier_counts[tier]
            for idx in sorted_indices[pos:pos + count]:
                blueprint[idx][col] = tier
            pos += count

    return blueprint


def format_tier_blueprint(
    blueprint: list[dict[str, str]],
    columns: Sequence[str],
) -> str:
    """Format tier blueprint as aggregate tier counts per column."""
    if not blueprint:
        return ""

    lines: list[str] = []
    for col in columns:
        counts: dict[str, int] = {}
        for row in blueprint:
            tier = row[col]
            counts[tier] = counts.get(tier, 0) + 1
        parts: list[str] = []
        for tier in ["LOW", "MID", "HIGH"]:
            if tier in counts:
                parts.append(f"{counts[tier]} {tier}")
        for tier, count in counts.items():
            if tier.startswith("SENTINEL"):
                parts.append(f"{count} {tier}")
        col_short = col.replace("expenditure_", "")
        lines.append(f"{col_short}: {', '.join(parts)}")

    return "\n".join(lines)


def format_correlation_text(
    pairs: list[tuple[str, str, float]],
    current_corr_map: dict[tuple[str, str], float] | None = None,
) -> str:
    if not pairs:
        return "No reliable correlations found in this dataset_segment."
    lines = []
    for a, b, val in pairs:
        interp = _interpret_correlation(val)

        cur_val: float | None = None
        if current_corr_map is not None:
            cur_val = current_corr_map.get((a, b)) or current_corr_map.get((b, a))

        if cur_val is not None:
            gap = abs(val - cur_val)
            if gap < 0.10:
                continue
            correction = "increase" if cur_val < val else "decrease"
            cur_str = f", current_r={cur_val:.3f}, gap={gap:.3f}, correction={correction}"
        else:
            cur_str = ", current_r=n/a"

        if abs(val) < 0.1:
            lines.append(
                f"{a} vs {b}: target_r={val:.3f}{cur_str} ({interp})."
            )
        else:
            direction = "positive" if val > 0 else "negative"
            lines.append(
                f"{a} vs {b}: target_r={val:.3f}{cur_str} ({interp}, {direction})."
            )
    return "\n".join(lines)
