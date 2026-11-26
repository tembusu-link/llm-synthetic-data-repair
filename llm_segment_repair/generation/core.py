"""
Core types, constants, and helpers for synthetic data generation.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Setting seed for reproducibility
np.random.seed(42)

# Directory holding the hand-authored synthetic texture-sample CSVs (one per
# dataset). The iterative generator reads its EXAMPLE ROWS block from these files
# instead of sampling real records, so generation runs at zero real-record
# exposure.
SAMPLE_DATA_DIR = Path(__file__).resolve().parent / "sample_data"


def sample_data_path_for(dataset_name: str) -> Path:
    """Resolve the synthetic texture-sample CSV for ``dataset_name``."""
    return SAMPLE_DATA_DIR / f"{dataset_name}.csv"


def _load_prompt_template_sections(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8")
    system_marker = "[[SYSTEM]]"
    user_marker = "[[USER]]"
    if system_marker not in text or user_marker not in text:
        raise ValueError(
            f"Prompt template file must contain {system_marker} and {user_marker}: {path}"
        )
    _, after_system = text.split(system_marker, 1)
    system_text, user_text = after_system.split(user_marker, 1)
    return path.stem, system_text.strip(), user_text.strip()


PROMPT_TEMPLATE_FILE = (
    Path(__file__).resolve().parent / "prompt_templates" / "v3.1_dataset_description.txt"
)
(
    PROMPT_TEMPLATE_VERSION,
    SYSTEM_PROMPT,
    TEMPLATE,
) = _load_prompt_template_sections(PROMPT_TEMPLATE_FILE)

ITERATIVE_PROMPT_TEMPLATE_FILE = (
    Path(__file__).resolve().parent / "prompt_templates" / "v3.2_dataset_level_targets.txt"
)
(
    ITERATIVE_PROMPT_TEMPLATE_VERSION,
    ITERATIVE_SYSTEM_PROMPT,
    ITERATIVE_TEMPLATE,
) = _load_prompt_template_sections(ITERATIVE_PROMPT_TEMPLATE_FILE)
ITERATIVE_PROMPT_TEMPLATE_FILE_V3_1 = (
    Path(__file__).resolve().parent / "prompt_templates" / "v3.1_dataset_description_iterative.txt"
)
(
    ITERATIVE_PROMPT_TEMPLATE_VERSION_V3_1,
    ITERATIVE_SYSTEM_PROMPT_V3_1,
    ITERATIVE_TEMPLATE_V3_1,
) = _load_prompt_template_sections(ITERATIVE_PROMPT_TEMPLATE_FILE_V3_1)

# Tier-allocation-free variant of the v3.2 template. Used when the ablation
# wants to suppress the per-column LOW/MID/HIGH/SENTINEL tier counts so that
# the only shape information reaching the LLM is the quantile anchors in the
# schema block.
ITERATIVE_PROMPT_TEMPLATE_FILE_NO_TIER = (
    Path(__file__).resolve().parent / "prompt_templates" / "v3.2_dataset_level_targets_no_tier.txt"
)
(
    ITERATIVE_PROMPT_TEMPLATE_VERSION_NO_TIER,
    ITERATIVE_SYSTEM_PROMPT_NO_TIER,
    ITERATIVE_TEMPLATE_NO_TIER,
) = _load_prompt_template_sections(ITERATIVE_PROMPT_TEMPLATE_FILE_NO_TIER)


@dataclass
class GenerationConfig:
    dataset_name: str
    baseline_variant: str
    synthetic_variant: str
    model: str
    temperature: float
    top_category_values: int
    sample_rows: int
    n_rows_multiplier: float
    max_rows_per_prompt: int
    progress_log: Path | None = None
    reasoning_effort: str | None = None
    categorical_features: Sequence[str] | None = None
    numeric_features: Sequence[str] | None = None
    correlation_columns: Sequence[str] | None = None
    api_key: str | None = None
    base_url: str | None = None
    overwrite: bool | None = None
    seed: int | None = None
    experiment_id: str | None = None
    ablation_label: str | None = None
    ablation_factors: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Provider-aware credential resolution
# ---------------------------------------------------------------------------

PROVIDER_ENV_MAP: dict[str, tuple[str, str]] = {
    "gemini": ("GEMINI_BASE_URL", "GEMINI_API_KEY"),
    "lmstudio": ("LMSTUDIO_BASE_URL", "LMSTUDIO_API_KEY"),
    "openai": ("OPENAI_BASE_URL", "OPENAI_API_KEY"),
}

PROVIDER_DEFAULTS: dict[str, str] = {
    "gemini": "gemini-2.0-flash",
    "lmstudio": "google/gemma-3-27b",
    "openai": "gpt-4o-mini",
}

MODEL_REASONING_DEFAULTS: dict[str, str] = {
    "gpt-5.2": "none",
}


def resolve_reasoning_effort(model: str, reasoning_effort: str | None) -> str | None:
    """Resolve configured reasoning effort and known model defaults."""
    if reasoning_effort is not None:
        return reasoning_effort
    return MODEL_REASONING_DEFAULTS.get(model)


def resolve_llm_credentials(
    provider: str | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str | None, str | None]:
    """Return ``(base_url, api_key)`` for the chosen provider.

    Explicit ``api_key`` / ``base_url`` arguments always win.  When they are
    ``None`` the function falls back to the provider-specific env vars.
    """
    if provider and provider not in PROVIDER_ENV_MAP:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDER_ENV_MAP)}"
        )

    if provider:
        url_var, key_var = PROVIDER_ENV_MAP[provider]
        base_url = base_url or os.getenv(url_var)
        api_key = api_key or os.getenv(key_var)
    else:
        # Legacy fallback chain: Gemini → LM Studio → OpenAI
        for _provider in ("gemini", "lmstudio", "openai"):
            url_var, key_var = PROVIDER_ENV_MAP[_provider]
            _url = os.getenv(url_var)
            _key = os.getenv(key_var)
            if _url:
                base_url = base_url or _url
                api_key = api_key or _key
                break

    return base_url, api_key


def _normalize_filter_value(value):
    if isinstance(value, str) and "[" in value and "]" in value:
        items = value.strip("[]").split(",")
        return [item.strip(" '\"") for item in items if item.strip(" '\"")]
    return value


def _to_builtin(value):
    if isinstance(value, np.generic):
        value = value.item()
    value = _normalize_filter_value(value)
    if isinstance(value, dict):
        return {key: _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(item) for item in value]
    return value


def _to_native_bool(value) -> bool | None:
    """Convert a value to a native Python bool, handling numpy types.

    MongoDB's BSON encoder cannot serialize numpy.bool_ types directly.
    This function ensures proper conversion to native Python bool.
    """
    if pd.isna(value):
        return None
    # Handle numpy scalar types by extracting the native Python value
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def readable_dataset_segment(meta: dict[str, object]) -> str:
    if not meta:
        return "Unknown dataset_segment (fallback to data patterns)"
    parts = []
    for key, value in meta.items():
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def format_json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower() or "model"


def build_synthetic_dataset_name(prefix: str, model: str) -> str:
    """Return a dataset name that embeds model identity for evaluation comparisons."""
    return f"{prefix}__{slugify(model)}"


def append_progress(
    progress_path: Path,
    dataset_segment_idx: int,
    dataset_segment_label: str,
    status: str,
    message: str = "",
) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "dataset_segment_index": dataset_segment_idx,
                "dataset_segment_label": dataset_segment_label,
                "status": status,
                "message": message,
            }
        ]
    )
    row.to_csv(progress_path, mode="a", header=not progress_path.exists(), index=False)


def load_completed(progress_path: Path) -> set[int]:
    if not progress_path.exists():
        return set()
    df = pd.read_csv(progress_path)
    return {int(val) for val in df[df["status"] == "completed"]["dataset_segment_index"].tolist()}


@dataclass
class IterativeGenerationConfig:
    """
    Configuration for iterative synthetic data generation.

    This approach generates data in multiple iterations, using feedback from
    the current synthetic data statistics to guide subsequent generations.

    Attributes:
        baseline_variant: Name of the real data dataset to use as target.
        synthetic_dataset_name: Name for the generated synthetic dataset.
        progress_log: Path to CSV file for tracking progress.
        model: LLM model identifier (e.g., "gpt-4o-mini").
        temperature: Sampling temperature for LLM.
        max_additional_iterations: Upper bound on iteration rounds per dataset_segment.
            The actual number of iterations is computed as
            ``ceil(rows_needed / rows_per_iteration)`` capped by this value.
        convergence_threshold: Mean absolute gap threshold to consider converged.
            Expressed as a fraction (e.g., 0.05 = 5% mean error).
        rows_per_iteration: Number of rows to generate in each iteration.
        top_n_gaps: Number of priority gaps to highlight in prompt.
        top_category_values: Max categories to show per categorical column.
        sample_rows: Number of example rows to include in prompt.
        sample_data_csv: Path to the hand-authored synthetic texture-sample CSV
            for this dataset. The EXAMPLE ROWS block is rendered from this file,
            never from real records, so generation runs at zero real-record
            exposure. The CSV must contain every schema column.
        n_rows_multiplier: Multiplier for target row count vs real data.
        categorical_features: List of categorical column names.
        numeric_features: List of numeric column names.
        correlation_columns: Optional list of columns for correlation tuning.
            If None, falls back to numeric_features.
        api_key: OpenAI API key.
        base_url: OpenAI API base URL.
        overwrite: If True, regenerate even if records exist.
    """

    dataset_name: str
    dataset_metadata: dict[str, object]
    baseline_variant: str
    synthetic_variant: str
    model: str
    temperature: float | None
    min_iterations: int
    max_additional_iterations: int
    convergence_threshold: float
    correlation_weight: float
    min_overall_score_delta: float
    max_frobenius_increase: float
    refinement_max_replace_fraction: float
    refinement_min_improvement: float
    repair_fraction: float
    repair_min_rows_per_iteration: int
    rows_per_iteration: int
    top_n_gaps: int
    top_category_values: int
    sample_rows: int
    n_rows_multiplier: float
    sample_data_csv: Path
    progress_log: Path | None = None
    num_percentiles: int | None = None
    # Over-request factor for each generation batch: the LLM often returns fewer
    # rows than asked, forcing extra top-up calls. Requesting
    # ceil(n * batch_buffer_factor) rows (excess truncated) lets a batch fill in
    # one shot. 1.0 = off (request exactly n); 1.1 = request 10% extra.
    batch_buffer_factor: float = 1.0
    # Lower bound on the row count requested from the LLM per prompt. The model
    # is unstable when asked for very few rows, so requests are floored here and
    # the surplus is truncated to the actual batch. 30 is the historical default;
    # set to 1 to make sub-30 batches a clean lever (e.g. learning-rate ablation).
    prompt_row_floor: int = 30
    include_current: bool = True
    disable_tier_allocation: bool = False
    disable_sentinels: bool = False
    reasoning_effort: str | None = None
    enforce_objective_acceptance: bool | None = None
    overwrite: bool | None = None
    seed: int | None = None
    segment_index: int | None = None
    segment_signature: str | None = None
    experiment_id: str | None = None
    ablation_label: str | None = None
    ablation_factors: dict[str, Any] | None = None
    segmenter_enabled: bool = False


@dataclass
class SentinelInfo:
    """
    Describes a point-mass sentinel detected at the extreme of a distribution.

    Attributes:
        value: The sentinel value (e.g. -1.0).
        mass: Estimated fraction of rows equal to this value.
        side: Which tail the sentinel sits on ("left" or "right").
    """

    value: float
    mass: float
    side: str


@dataclass
class NumericTextureProfile:
    """
    Universal numeric column profile for shape-based synthesis.

    This is dataset-agnostic and captures core distribution shape.
    When sentinels are detected (flat quantile regions at distribution
    extremes), the quantiles and support are recomputed on the
    continuous portion only, and the sentinel info is stored separately.

    Attributes:
        column: Column name
        min: Minimum value (of continuous portion when sentinels present)
        max: Maximum value (of continuous portion when sentinels present)
        p_zero: Fraction of exact zeros
        quantiles: Sparse quantile profile (p1..p99) over the continuous portion
        sentinels: Detected point-mass sentinels at distribution extremes
    """

    column: str
    min: float
    max: float
    p_zero: float
    quantiles: dict[str, float]
    sentinels: list[SentinelInfo] | None = None


@dataclass
class GapDirective:
    """
    A universal directive describing a gap to correct.

    This is dataset-agnostic and can express gaps in:
    - Quantile positions (shape)
    - Texture metrics (decimal patterns, gridness)
    - Spike/duplicate suppression
    - Correlation/dependence

    Attributes:
        priority: Importance rank (1 = highest priority)
        gap_type: Type of gap (quantile_gap, texture_gap, spike_suppression, corr_gap, etc.)
        column: Column name (or list of columns for correlation gaps)
        metric: Specific metric within the type (e.g., "p90" for quantile, "fractional_topk")
        direction: Correction direction (increase, decrease, match_target, reduce, etc.)
        target_value: Target value if applicable
        current_value: Current value if applicable
        note: Human-readable explanation
    """

    priority: int
    gap_type: str
    column: str | list[str]
    metric: str
    direction: str
    target_value: float | str | None = None
    current_value: float | str | None = None
    note: str | None = None
    score_before: float | None = None
    score_after: float | None = None
    # Per-cell breakdown for directives that aggregate many cells under a
    # single column-level score (e.g. debug quantile_gap: one directive per
    # column, six percentiles packed as cells). Each cell:
    # {"metric": str, "target": float, "current": float, "value_after": float | None}.
    cells: list[dict] | None = None


@dataclass
class GapMetric:
    """
    Represents a gap between synthetic and target statistics.

    Attributes:
        attribute: Column name.
        metric_type: Type of metric (e.g., "categorical_freq", "numeric_mean").
        target_value: Target value from real data.
        current_value: Current value from synthetic data.
        gap: Absolute difference (target - current).
        direction: "increase" or "decrease" to indicate correction direction.
        priority_score: Score for ranking (higher = more important to fix).
    """

    attribute: str
    metric_type: str
    target_value: float
    current_value: float
    gap: float
    direction: str
    priority_score: float
    detail: str | None = None
