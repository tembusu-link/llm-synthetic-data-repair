"""LLM generators and prompt assembly."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict
from functools import cache
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from scipy.stats import chi2_contingency


from .core import (
    ITERATIVE_SYSTEM_PROMPT,
    ITERATIVE_SYSTEM_PROMPT_NO_TIER,
    ITERATIVE_TEMPLATE,
    ITERATIVE_TEMPLATE_NO_TIER,
    IterativeGenerationConfig,
    _to_builtin,
    readable_dataset_segment,
    resolve_reasoning_effort,
)
from .gaps import (
    compute_debug_gap_directives,
    compute_gap,
    compute_gap_directives,
    format_gap_directives,
)
from .schema import ResolvedSchema, resolve_dataset_schema
from .stats import (
    compute_all_texture_profiles,
    compute_tier_blueprint,
    format_correlation_text,
    format_tier_blueprint,
    generate_schema_block,
    summarise_categorical,
    summarise_correlations,
)
from .structured import build_structured_output_models

logger = logging.getLogger(__name__)


def _serialize_texture_profiles(
    profiles: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Convert numeric texture profile dataclasses to plain dictionaries."""
    serialized: dict[str, dict[str, object]] = {}
    for col, profile in profiles.items():
        if hasattr(profile, "__dataclass_fields__"):
            serialized[col] = _to_builtin(asdict(profile))
        else:
            serialized[col] = _to_builtin(profile)
    return serialized


def build_target_statistics(
    *,
    target_df: pd.DataFrame,
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
    corr_cols: Sequence[str],
    top_category_values: int,
) -> dict[str, object]:
    """Build a reusable target-statistics payload for iteration metadata."""
    categorical_summary = summarise_categorical(target_df, cat_cols, top_category_values)
    texture_profiles = compute_all_texture_profiles(target_df, num_cols)
    correlation_summary = summarise_correlations(target_df, corr_cols)
    return {
        "categorical_summary": _to_builtin(categorical_summary),
        "numeric_texture_profiles": _serialize_texture_profiles(texture_profiles),
        "correlation_summary": _to_builtin(correlation_summary),
    }


def parse_model_output(result: object, expected_columns: Sequence[str]) -> pd.DataFrame:
    """Convert structured LLM output to a DataFrame.

    Args:
        result: Pydantic model returned by the LLM via structured output.
        expected_columns: List of column names to keep (in order).

    Returns:
        DataFrame with valid rows only.

    Raises:
        ValueError: If no rows or required columns missing.
    """
    rows = getattr(result, "rows", None)
    if not rows:
        raise ValueError("Model output contains no rows")

    df = pd.DataFrame([row.model_dump() for row in rows])

    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns from model output: {missing}")

    return df[list(expected_columns)]


def invoke_structured_rows_prompt(
    *,
    llm: ChatOpenAI,
    prompt: ChatPromptTemplate,
    payload: dict[str, object],
    rows_model: type,
    expected_columns: Sequence[str],
    expected_n_rows: int | None = None,
    max_attempts: int = 3,
    batch_buffer_factor: float = 1.0,
    prompt_row_floor: int = 30,
    before_call: Callable[[], None],
) -> pd.DataFrame:
    """Invoke a structured-output row prompt with the shared retry policy."""
    structured_llm = llm.with_structured_output(rows_model)

    def invoke(messages):
        before_call()
        return structured_llm.invoke(messages)
    if expected_n_rows is None:
        prompt_payload = dict(payload)
        if "n_rows" in prompt_payload:
            prompt_payload["n_rows"] = max(prompt_row_floor, int(prompt_payload["n_rows"]))
        input = prompt.format_messages(**prompt_payload)
        response = invoke(input)
        return parse_model_output(response, expected_columns=expected_columns)

    frames: list[pd.DataFrame] = []
    rows_remaining = int(expected_n_rows)
    attempt = 0
    consecutive_failures = 0
    while rows_remaining > 0 and consecutive_failures < max_attempts:
        attempt += 1
        attempt_payload = dict(payload)
        attempt_payload["n_rows"] = max(
            prompt_row_floor, math.ceil(rows_remaining * batch_buffer_factor)
        )
        input = prompt.format_messages(**attempt_payload)
        response = invoke(input)
        try:
            batch_df = parse_model_output(response, expected_columns=expected_columns)
        except ValueError as exc:
            assert str(exc) == "Model output contains no rows", str(exc)
            batch_df = pd.DataFrame(columns=expected_columns)
        if len(batch_df) > rows_remaining:
            batch_df = batch_df.head(rows_remaining)
        received_rows = len(batch_df)
        if received_rows == 0:
            consecutive_failures += 1
        else:
            frames.append(batch_df)
            rows_remaining -= received_rows
            consecutive_failures = 0
        logger.warning(
            "structured_output_underfilled attempt=%d expected=%d received=%d remaining=%d consecutive_failures=%d",
            attempt,
            expected_n_rows,
            received_rows,
            rows_remaining,
            consecutive_failures,
        )

    non_empty_frames = [frame for frame in frames if not frame.empty]
    combined = (
        pd.concat(non_empty_frames, ignore_index=True)
        if non_empty_frames
        else pd.DataFrame(columns=expected_columns)
    )
    if len(combined) < expected_n_rows:
        logger.warning(
            "structured_output_underfilled_final expected=%d got=%d after_attempts=%d",
            expected_n_rows,
            len(combined),
            attempt,
        )
        raise AssertionError(
            f"LLM structured output underfilled: expected {expected_n_rows} rows, "
            f"got {len(combined)} after {attempt} attempts "
            f"({max_attempts} consecutive no-row failures)"
        )
    assert len(combined) == expected_n_rows, (
        f"LLM structured output row-count mismatch: expected {expected_n_rows}, got {len(combined)}"
    )
    return combined


def _df_to_json_rows(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a JSON array string, one object per line."""
    records = json.loads(df.to_json(orient="records"))
    return json.dumps(records, indent=None)


@cache
def _load_sample_rows_cached(csv_path: str, columns: tuple[str, ...]) -> str:
    df = pd.read_csv(csv_path)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"sample_data CSV {csv_path!r} is missing required schema columns: "
            f"{missing}. The synthetic texture sample must contain every schema "
            "column."
        )
    return _df_to_json_rows(df[list(columns)])


def load_sample_rows(csv_path: Path, columns: Sequence[str]) -> str:
    """Render the hand-authored synthetic texture sample as JSON example rows.

    The EXAMPLE ROWS block shown to the LLM is read from a fixed, per-dataset
    synthetic CSV (cached after first read) rather than sampled from real
    records, so the generator never sees a real row (zero real-record exposure).
    Only the requested schema ``columns`` are emitted; the function raises if the
    CSV is missing any of them.
    """
    return _load_sample_rows_cached(str(csv_path), tuple(columns))


def example_rows_text(df: pd.DataFrame, columns: Sequence[str], sample_rows: int) -> str:
    """Select example rows that exhibit diverse text profiles via maxmin diversity sampling.

    Used by real-data consumers such as incoherence detection and the EPIC
    few-shot baseline. The iterative generator does NOT use this; it reads a
    fixed synthetic texture sample via :func:`load_sample_rows` instead, to keep
    real-record exposure at zero.
    """
    keep_cols = [col for col in columns if col in df.columns]
    pool = df[keep_cols]

    if len(pool) <= sample_rows:
        return _df_to_json_rows(pool)

    # Subsample candidates for efficiency on very large frames
    max_candidates = 2000
    if len(pool) > max_candidates:
        pool = pool.sample(n=max_candidates, random_state=42)
    pool = pool.reset_index(drop=True)
    n = len(pool)

    # Build feature vectors that capture diversity without hard-coded rules.
    # - String length of each value's representation captures text profile
    #   (magnitude, decimal places, formatting) in a generic way.
    # - Normalised numeric value captures value-range spread.
    features: list[np.ndarray] = []
    for col in keep_cols:
        s = pool[col]
        str_lens = s.astype(str).str.len().values.astype(float)
        std = str_lens.std()
        if std > 0:
            features.append((str_lens - str_lens.mean()) / std)

        if pd.api.types.is_numeric_dtype(s):
            # Convert boolean to int (0/1) to allow arithmetic operations
            if s.dtype == bool or pd.api.types.is_bool_dtype(s):
                s = s.astype(int)
            mn, mx = s.min(), s.max()
            if mx > mn:
                features.append(((s - mn) / (mx - mn)).fillna(0).values)
        else:
            # For categorical columns, use normalised category codes so
            # different categories are spread apart.
            codes = s.astype("category").cat.codes.values.astype(float)
            mx = codes.max()
            if mx > 0:
                features.append(codes / mx)

    if not features:
        return _df_to_json_rows(pool.head(sample_rows))

    feat_matrix = np.column_stack(features)

    # Greedy farthest-point (maxmin) diversity selection
    selected = [0]
    diffs = feat_matrix - feat_matrix[0]
    min_dists = np.einsum("ij,ij->i", diffs, diffs)  # squared L2
    min_dists[0] = -1.0

    for _ in range(sample_rows - 1):
        best = int(np.argmax(min_dists))
        if min_dists[best] <= 0:
            break
        selected.append(best)
        min_dists[best] = -1.0
        diffs = feat_matrix - feat_matrix[best]
        new_dists = np.einsum("ij,ij->i", diffs, diffs)
        mask = min_dists >= 0
        min_dists = np.where(mask, np.minimum(min_dists, new_dists), min_dists)

    # Fill remaining slots if diversity was exhausted early
    if len(selected) < sample_rows:
        chosen = set(selected)
        for i in range(n):
            if len(selected) >= sample_rows:
                break
            if i not in chosen:
                selected.append(i)

    return _df_to_json_rows(pool.iloc[sorted(selected)])


def _target_interval_specs(profile) -> list[tuple[str, float, float, float]]:
    """Return quantile-derived interval specs with nominal target probabilities.

    When sentinels are present the continuous-portion probabilities are
    scaled by ``(1 - total_sentinel_mass)`` and a dedicated sentinel
    interval is prepended (left) or appended (right).
    """
    sentinels = getattr(profile, "sentinels", None) or []
    total_sentinel_mass = sum(s.mass for s in sentinels)
    cont_scale = max(1.0 - total_sentinel_mass, 0.0)

    quantiles = profile.quantiles

    # Base continuous intervals (probabilities are fractions of the
    # *continuous* portion of the distribution).
    raw_intervals = [
        ("[min,p1]", float(profile.min), float(quantiles["p1"]), 0.01),
        ("(p1,p5]", float(quantiles["p1"]), float(quantiles["p5"]), 0.04),
        ("(p5,p10]", float(quantiles["p5"]), float(quantiles["p10"]), 0.05),
        ("(p10,p25]", float(quantiles["p10"]), float(quantiles["p25"]), 0.15),
        ("(p25,p50]", float(quantiles["p25"]), float(quantiles["p50"]), 0.25),
        ("(p50,p75]", float(quantiles["p50"]), float(quantiles["p75"]), 0.25),
        ("(p75,p90]", float(quantiles["p75"]), float(quantiles["p90"]), 0.15),
        ("(p90,p95]", float(quantiles["p90"]), float(quantiles["p95"]), 0.05),
        ("(p95,p99]", float(quantiles["p95"]), float(quantiles["p99"]), 0.04),
        ("(p99,max]", float(quantiles["p99"]), float(profile.max), 0.01),
    ]

    # Scale continuous probabilities into overall-distribution space
    scaled: list[tuple[str, float, float, float]] = [
        (label, lo, hi, round(prob * cont_scale, 4)) for label, lo, hi, prob in raw_intervals
    ]

    # Inject sentinel intervals
    for s in sentinels:
        sentinel_label = f"[sentinel={s.value}]"
        if s.side == "left":
            scaled.insert(0, (sentinel_label, s.value, s.value, round(s.mass, 4)))
        else:
            scaled.append((sentinel_label, s.value, s.value, round(s.mass, 4)))

    return scaled


def _count_interval(series: pd.Series, lower: float, upper: float, interval_idx: int) -> int:
    """Count values falling in one quantile-derived interval.

    Sentinel intervals have ``lower == upper`` (point mass); use exact
    equality for those.
    """
    if series.empty:
        return 0
    if lower == upper:
        # Point-mass / sentinel interval
        mask = series == lower
    elif interval_idx == 0:
        mask = (series >= lower) & (series <= upper)
    else:
        mask = (series > lower) & (series <= upper)
    return int(mask.sum())


def format_numeric_correction_summary(
    target_profiles: dict[str, object],
    current_df: pd.DataFrame | None,
    numerical_columns: Sequence[str],
    batch_n_rows: int,
) -> str:
    """Format numeric correction targets for the current iteration batch."""
    if not numerical_columns:
        return "No numeric columns available for correction targets."

    current_total_rows = 0 if current_df is None else len(current_df)
    final_total_rows = current_total_rows + batch_n_rows
    lines: list[str] = [
        (
            f"This batch adds {batch_n_rows} rows. Current synthetic rows: {current_total_rows}. "
            f"Target final rows after this batch: {final_total_rows}."
        )
    ]

    for col in numerical_columns:
        profile = target_profiles.get(col)
        if profile is None:
            continue

        target_median = float(profile.quantiles["p50"])
        current_series = pd.Series(dtype=float)
        current_mean_text = "n/a"
        mean_delta_text = "n/a"
        if current_df is not None and col in current_df.columns and not current_df.empty:
            current_series = pd.to_numeric(current_df[col], errors="coerce").dropna()
            if not current_series.empty:
                current_mean = float(current_series.mean())
                current_mean_text = f"{current_mean:.3f}"
                mean_delta = target_median - current_mean
                mean_delta_text = f"{mean_delta:+.3f}"

        sentinel_note = ""
        sentinels = getattr(profile, "sentinels", None) or []
        if sentinels:
            parts = [f"{s.value}@{round(s.mass * 100, 1)}%" for s in sentinels]
            sentinel_note = f", sentinels=[{', '.join(parts)}]"

        lines.append(
            f"\n{col}: target_median_anchor={target_median:.3f}, "
            f"current_mean={current_mean_text}, signed_mean_delta={mean_delta_text}, "
            f"p_zero_target={float(profile.p_zero):.4f}{sentinel_note}"
        )

        for idx, (label, lower, upper, prob) in enumerate(_target_interval_specs(profile)):
            target_final = int(round(prob * final_total_rows))
            current_count = _count_interval(current_series, lower, upper, idx)
            deficit = target_final - current_count
            batch_target = max(deficit, 0)
            lines.append(
                f"  - {label}: target_prob={prob:.2f}, target_final_count~{target_final}, "
                f"current_count={current_count}, batch_deficit={deficit:+d}, "
                f"suggested_batch_fill={batch_target}"
            )

    return "\n".join(lines)


def format_correlation_gap_summary(
    target_df: pd.DataFrame,
    current_df: pd.DataFrame | None,
    corr_columns: Sequence[str],
    top_k: int = 6,
) -> str:
    """Format current-vs-target correlation errors for prompt steering."""
    if current_df is None or current_df.empty or len(current_df) < 3:
        return "No reliable current synthetic correlation matrix yet. Use target correlations as the initial plan."

    usable_cols = [
        col for col in corr_columns if col in target_df.columns and col in current_df.columns
    ]
    if len(usable_cols) < 2:
        return "No reliable correlation columns available for current-vs-target comparison."

    target_corr = target_df[usable_cols].apply(pd.to_numeric, errors="coerce").corr()
    current_corr = current_df[usable_cols].apply(pd.to_numeric, errors="coerce").corr()
    common_cols = [
        col for col in usable_cols if col in target_corr.index and col in current_corr.index
    ]
    if len(common_cols) < 2:
        return "No reliable current-vs-target correlation comparison available."

    diff = target_corr.loc[common_cols, common_cols] - current_corr.loc[common_cols, common_cols]
    fro_gap = float(np.linalg.norm(diff.fillna(0.0).values, ord="fro"))

    lines = [f"Current correlation Frobenius gap: {fro_gap:.4f}"]
    pair_gaps: list[tuple[str, str, float, float, float]] = []
    for idx, col_a in enumerate(common_cols):
        for col_b in common_cols[idx + 1 :]:
            target_val = target_corr.loc[col_a, col_b]
            current_val = current_corr.loc[col_a, col_b]
            if pd.isna(target_val) or pd.isna(current_val):
                continue
            pair_gaps.append(
                (
                    col_a,
                    col_b,
                    float(target_val),
                    float(current_val),
                    abs(float(target_val) - float(current_val)),
                )
            )

    if not pair_gaps:
        lines.append("No reliable pairwise correlation gaps available.")
        return "\n".join(lines)

    pair_gaps.sort(key=lambda item: item[4], reverse=True)
    for col_a, col_b, target_val, current_val, gap in pair_gaps[:top_k]:
        direction = "increase" if current_val < target_val else "decrease"
        lines.append(
            f"- {col_a} vs {col_b}: target_r={target_val:.3f}, current_r={current_val:.3f}, "
            f"abs_gap={gap:.3f}, correction={direction}"
        )

    return "\n".join(lines)


class IterativeLLMDatasetSegmentGeneratorBase:
    """
    Iterative synthetic data generator using LLM with gap feedback.

    This generator implements an iterative refinement approach:
    1. Generate initial batch of synthetic rows
    2. Compute statistics and identify gaps vs target
    3. Generate additional rows with prompt highlighting top gaps
    4. Repeat until convergence or max iterations

    The key insight is that by showing the LLM the current gaps, it can
    generate rows that specifically help close those gaps, leading to
    better statistical fidelity than single-shot generation.

    Attributes:
        config: IterativeGenerationConfig with generation parameters.
        llm: LangChain LLM instance.
        chain: LangChain chain for prompt-to-response.
    """

    def __init__(self, config: IterativeGenerationConfig, *, max_calls: int):
        """
        Initialize the iterative generator.

        Args:
            config: Configuration object with generation parameters.
            max_calls: Total model invocations allowed across all segments.
        """
        assert max_calls > 0
        self.config = config
        self.max_calls = max_calls
        self.calls_used = 0

        if config.seed is not None:
            np.random.seed(config.seed)
        self.schema = resolve_dataset_schema(
            dataset_name=self.config.dataset_name,
            dataset_metadata=self.config.dataset_metadata,
        )
        self._structured_models: dict[
            tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]],
            tuple[type, type],
        ] = {}

        resolved_reasoning_effort = resolve_reasoning_effort(config.model, config.reasoning_effort)

        logger.info("Initializing IterativeLLMDatasetSegmentGenerator")
        logger.info(f"  min_iterations: {config.min_iterations}")
        logger.info(f"  max_additional_iterations: {config.max_additional_iterations}")
        logger.info(f"  convergence_threshold: {config.convergence_threshold}")
        logger.info(f"  rows_per_iteration: {config.rows_per_iteration}")
        logger.info(f"  reasoning_effort: {resolved_reasoning_effort}")

        self.llm = ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            reasoning_effort=resolved_reasoning_effort,
            max_retries=0,
        )

        if config.disable_tier_allocation:
            system_prompt = ITERATIVE_SYSTEM_PROMPT_NO_TIER
            user_template = ITERATIVE_TEMPLATE_NO_TIER
            logger.info("  disable_tier_allocation: True (using no-tier prompt template)")
        else:
            system_prompt = ITERATIVE_SYSTEM_PROMPT
            user_template = ITERATIVE_TEMPLATE
        self._system_prompt_text = system_prompt
        self._user_template_text = user_template
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", user_template)]
        )

    def _get_structured_models(self, resolved) -> tuple[type, type]:
        key = (
            tuple(resolved.columns),
            tuple(resolved.boolean),
            tuple(resolved.integer),
            tuple(resolved.numerical),
        )
        cached = self._structured_models.get(key)
        if cached:
            return cached
        row_model, rows_model = build_structured_output_models(resolved)
        self._structured_models[key] = (row_model, rows_model)
        return row_model, rows_model

    def _reserve_call(self) -> None:
        if self.calls_used >= self.max_calls:
            raise RuntimeError(
                f"Stopped after {self.max_calls} LLM calls. Request budget exhausted."
            )
        self.calls_used += 1

    def _invoke_structured_rows(
        self,
        *,
        payload: dict[str, object],
        rows_model: type,
        expected_columns: Sequence[str],
        expected_n_rows: int | None = None,
        max_attempts: int = 3,
        batch_buffer_factor: float = 1.0,
        prompt_row_floor: int = 30,
    ) -> pd.DataFrame:
        """Invoke the iterative prompt and parse structured row output.

        Every LLM prompt requests at least 30 rows. When ``expected_n_rows`` is
        provided, keep requesting the remaining row count until complete or
        until ``max_attempts`` consecutive no-row responses occur.
        ``batch_buffer_factor`` multiplies each attempt's requested count so the
        model over-generates; excess rows are truncated to the actual shortfall.
        """
        return invoke_structured_rows_prompt(
            llm=self.llm,
            prompt=self.prompt,
            payload=payload,
            rows_model=rows_model,
            expected_columns=expected_columns,
            expected_n_rows=expected_n_rows,
            max_attempts=max_attempts,
            batch_buffer_factor=batch_buffer_factor,
            prompt_row_floor=prompt_row_floor,
            before_call=self._reserve_call,
        )

    @staticmethod
    def _compute_contingency_pairs(
        df: pd.DataFrame,
        categorical_cols: list[str],
        min_association: float = 0.3,
        top_cells: int = 3,
    ) -> list[dict[str, object]]:
        """Compute Cramér's V association strength for categorical column pairs from real data.

        Measures the strength of association between categorical columns using chi-square
        test. Only includes pairs with association >= min_association. Cramér's V ranges
        from 0 (independent) to 1 (perfect association).

        For each kept pair, also records the top ``top_cells`` most-frequent
        (c1_val, c2_val) joint combinations with their joint probabilities so
        the schema block can show *which* values tend to co-occur, not just
        how strong the association is.

        Args:
            df: DataFrame with categorical data.
            categorical_cols: Column names to analyze.
            min_association: Minimum Cramér's V to include (default 0.3).
            top_cells: How many top joint combinations to record per pair.

        Returns:
            List of dicts with c1, c2, score, top_pairs, and diagnostic fields.
        """
        cat_cols_present = [c for c in categorical_cols if c in df.columns]
        if len(cat_cols_present) < 2:
            return []

        contingency_pairs = []
        for c1, c2 in combinations(cat_cols_present, 2):
            try:
                ct = pd.crosstab(df[c1], df[c2])
                chi2, p_value, dof, expected = chi2_contingency(ct)

                n = ct.sum().sum()
                min_dim = min(ct.shape) - 1
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0

                if cramers_v >= min_association:
                    top_pairs = []
                    if n > 0:
                        stacked = ct.stack().sort_values(ascending=False).head(top_cells)
                        for idx, count in stacked.items():
                            top_pairs.append(
                                {
                                    "c1_val": str(idx[0]),
                                    "c2_val": str(idx[1]),
                                    "freq": float(count) / float(n),
                                }
                            )
                    contingency_pairs.append(
                        {
                            "c1": c1,
                            "c2": c2,
                            "score": float(cramers_v),
                            "key": f"{c1}__{c2}",
                            "chi2": float(chi2),
                            "p_value": float(p_value),
                            "top_pairs": top_pairs,
                        }
                    )
            except Exception as e:
                logger.debug(f"Contingency computation skipped for {c1} vs {c2}: {e}")
                continue

        return contingency_pairs

    def _build_iterative_prompt_payload(
        self,
        target_df: pd.DataFrame,
        current_synthetic_df: pd.DataFrame | None,
        schema: ResolvedSchema,
        n_rows: int,
        iteration_num: int,
        target_total_rows: int | None = None,
        total_iterations: int | None = None,
        current_report: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """
        Build prompt payload with gap feedback for iterative generation.

        Uses dataset-agnostic NumericTextureProfiles and GapDirectives
        instead of domain-specific statistics.

        Args:
            target_df: Real data (target distribution).
            current_synthetic_df: Current synthetic data (None for first iteration).
            schema: Resolved dataset schema.
            n_rows: Number of rows to generate this iteration.
            iteration_num: Current iteration number.
            target_total_rows: Total rows the final synthetic dataset should have.
            total_iterations: Total planned iterations for this dataset segment.

        Returns:
            Dictionary of prompt template variables.
        """
        # Target statistics (from real data)
        categorical_summary = summarise_categorical(
            target_df, schema.categorical, self.config.top_category_values
        )
        # Universal texture profiles for schema block + gap directives
        target_texture_profiles = compute_all_texture_profiles(
            target_df,
            schema.numerical,
            disable_sentinels=self.config.disable_sentinels,
        )
        correlations = summarise_correlations(target_df, schema.correlation)

        # Compute target contingency pairs (categorical relationships)
        target_contingency_pairs = self._compute_contingency_pairs(target_df, schema.categorical)

        # Current synthetic statistics and gap directives
        current_categorical_summary: dict[str, dict[str, float]] | None = None
        current_corr_map: dict[tuple[str, str], float] | None = None
        current_contingency_pairs: list[dict[str, object]] | None = None

        # Two prompt channels are derived from the current estimate and are
        # toggled independently (the information-layer ablation is a 2x2 over
        # these factors):
        #   * include_current  — whether the RAW current statistics (quantiles,
        #     marginals, correlations) are displayed beside the target in the
        #     schema/correlation blocks (Channels A/B).
        #   * top_n_gaps > 0   — whether the PRE-DIGESTED gap directives are
        #     emitted (Channel C). Directives are derived from the
        #     current-vs-target gap, but are shown independently of whether the
        #     raw current statistics are displayed.
        # The current estimate is therefore always profiled when it exists; the
        # two switches only decide what reaches the prompt.
        has_current = current_synthetic_df is not None and len(current_synthetic_df) > 0
        current_row_count = len(current_synthetic_df) if has_current else 0

        if has_current:
            current_texture_profiles_full = compute_all_texture_profiles(
                current_synthetic_df,
                schema.numerical,
                target_profiles=target_texture_profiles,
                disable_sentinels=self.config.disable_sentinels,
            )
            current_categorical_summary_full = summarise_categorical(
                current_synthetic_df, schema.categorical, self.config.top_category_values
            )
            current_corr_pairs = summarise_correlations(current_synthetic_df, schema.correlation)
            current_corr_map_full = {(a, b): v for a, b, v in current_corr_pairs}
            current_contingency_pairs = self._compute_contingency_pairs(
                current_synthetic_df, schema.categorical
            )
        else:
            current_texture_profiles_full = {}
            current_categorical_summary_full = None
            current_corr_map_full = None

        # Channel C: pre-digested gap directives (gated by top_n_gaps).
        if has_current and self.config.top_n_gaps > 0:
            gap_directives = compute_gap_directives(
                target_profiles=target_texture_profiles,
                current_profiles=current_texture_profiles_full,
                target_df=target_df,
                current_df=current_synthetic_df,
                dataset_metadata=self.config.dataset_metadata,
                schema=schema,
                target_contingency_pairs=target_contingency_pairs,
                current_contingency_pairs=current_contingency_pairs,
                max_directives=self.config.top_n_gaps,
                report=current_report,
            )
            # Suppress per-column numerical directives ONLY for columns a
            # corr_gap already steers — a corr_gap dictates the direction of
            # both columns in the pair, so a coincident quantile/sentinel gap
            # on either of those columns gives the LLM conflicting guidance.
            # Columns not in any corr_gap keep their shape directives.
            corr_steered_columns: set[str] = set()
            for d in gap_directives:
                if d.gap_type == "corr_gap" and isinstance(d.column, list):
                    corr_steered_columns.update(str(c) for c in d.column)

            def _suppressed_by_corr(d) -> bool:
                if d.gap_type not in {"quantile_gap", "sentinel_gap"}:
                    return False
                col = str(d.column) if isinstance(d.column, str) else None
                return col in corr_steered_columns

            gap_directives = [d for d in gap_directives if not _suppressed_by_corr(d)]
            for i, d in enumerate(gap_directives):
                d.priority = i + 1
            gap_directives_text = format_gap_directives(
                gap_directives,
                n_rows=n_rows,
                current_row_count=current_row_count,
                target_total_rows=target_total_rows if isinstance(target_total_rows, int) else 0,
            )
            gap_directives_raw = gap_directives

            # Debug directives: full directive set for low-quality columns/pairs.
            # Persisted for post-hoc debugging; not formatted into the prompt.
            if current_report is not None:
                debug_gap_directives_raw = compute_debug_gap_directives(
                    report=current_report,
                    schema=schema,
                    target_contingency_pairs=target_contingency_pairs,
                    target_profiles=target_texture_profiles,
                    current_profiles=current_texture_profiles_full,
                )
            else:
                debug_gap_directives_raw = []
        else:
            gap_directives_raw = []
            debug_gap_directives_raw = []
            gap_directives_text = (
                "No significant gaps detected." if has_current else "First iteration — no gaps yet."
            )

        # Channels A/B: raw current-statistics display (gated by include_current).
        if self.config.include_current:
            current_texture_profiles = current_texture_profiles_full
            current_categorical_summary = current_categorical_summary_full
            current_corr_map = current_corr_map_full
        else:
            current_texture_profiles = {}
            current_categorical_summary = None
            current_corr_map = None

        # Null rates per column (from real data)
        null_rates = {
            col: float(target_df[col].isna().mean())
            for col in schema.columns
            if col in target_df.columns
        }

        # Generate schema block (enriched with target + current profiles).
        # Contingency partners are inlined under each categorical column here;
        # the standalone CATEGORICAL PAIR CONTINGENCY prompt section is gone.
        schema_block = generate_schema_block(
            schema,
            categorical_summary,
            target_texture_profiles,
            null_rates,
            current_categorical_summary,
            current_texture_profiles or None,
            num_percentiles=self.config.num_percentiles,
        )
        correlation_text = format_correlation_text(correlations, current_corr_map)

        # Build constraints block (dataset-agnostic, derived from schema)
        constraints_lines = [
            "- Categorical values must exactly match one of the categories listed in the schema.",
            "- For categorical columns, use only the original allowed category values.",
            "- If 'Other' appears in a category list, emit exactly 'Other' (no variants).",
            "- Boolean fields must be exactly True or False.",
            "- Provide a value for every field with a zero target null rate.",
            "- Numeric values must fall within the support range specified in the profile.",
        ]
        constraints_block = "\n".join(constraints_lines)

        dataset_segment_label = readable_dataset_segment(schema.metadata_dict)

        _target_total_rows = target_total_rows if target_total_rows is not None else "n/a"
        _total_iterations = total_iterations if total_iterations is not None else "n/a"

        # Pre-compute tier assignments from target correlations,
        # sentinel mass, and exact tier counts to prevent overshoot.
        # Skipped entirely when tier allocation is disabled — the no-tier
        # template has no {tier_blueprint} placeholder.
        if self.config.disable_tier_allocation:
            tier_blueprint = {}
            tier_blueprint_text = ""
        else:
            tier_blueprint = compute_tier_blueprint(
                n_rows=n_rows,
                columns=schema.numerical,
                target_corr_pairs=correlations,
                target_profiles=target_texture_profiles,
                current_profiles=current_texture_profiles or None,
                current_row_count=current_row_count,
                current_df=current_synthetic_df,
            )
            tier_blueprint_text = format_tier_blueprint(tier_blueprint, schema.numerical)

        template_kwargs = dict(
            dataset_name=self.config.dataset_name,
            dataset_description=self.schema.dataset_description,
            dataset_segment_label=dataset_segment_label,
            schema_block=schema_block,
            constraints_block=constraints_block,
            correlation_summary=correlation_text,
            current_row_count=current_row_count,
            iteration_num=iteration_num,
            gap_directives=gap_directives_text,
            example_rows=load_sample_rows(self.config.sample_data_csv, schema.columns),
            n_rows=n_rows,
            target_total_rows=_target_total_rows,
            total_iterations=_total_iterations,
        )
        if not self.config.disable_tier_allocation:
            template_kwargs["tier_blueprint"] = tier_blueprint_text
        prompt_text = self._user_template_text.format(**template_kwargs)

        logger.info(
            "Iterative prompt (iteration %s, n_rows=%s):\nSYSTEM:\n%s\n\nHUMAN:\n%s",
            iteration_num,
            n_rows,
            self._system_prompt_text,
            prompt_text,
        )

        payload: dict[str, object] = {
            "dataset_name": self.config.dataset_name,
            "dataset_segment_label": dataset_segment_label,
            "dataset_description": self.schema.dataset_description,
            "schema_block": schema_block,
            "constraints_block": constraints_block,
            "correlation_summary": correlation_text,
            "current_row_count": current_row_count,
            "iteration_num": iteration_num,
            "gap_directives": gap_directives_text,
            "example_rows": load_sample_rows(self.config.sample_data_csv, schema.columns),
            "n_rows": n_rows,
            "target_total_rows": _target_total_rows,
            "total_iterations": _total_iterations,
            "prompt_text": f"SYSTEM:\n{self._system_prompt_text}\n\nHUMAN:\n{prompt_text}",
            # Raw directives for post-iteration analysis (popped before chain.invoke)
            "_gap_directives_raw": gap_directives_raw,
            "_debug_gap_directives_raw": debug_gap_directives_raw,
            "_target_texture_profiles": target_texture_profiles,
        }
        if not self.config.disable_tier_allocation:
            payload["tier_blueprint"] = tier_blueprint_text
        return payload
