"""Dataset metadata loading and schema helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _metadata_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def metadata_path_for(dataset_name: str) -> Path:
    return _metadata_dir() / f"{dataset_name}.json"


def segment_filter_cache_path(
    dataset_name: str, baseline_variant: str = "real_data"
) -> Path:
    """Return path to the segment filter cache file."""
    cache_name = f"{dataset_name}__{baseline_variant}__segment_filters.json"
    return _metadata_dir() / cache_name


def load_metadata_dict(dataset_name: str, *, path: Path | None = None) -> dict[str, Any]:
    metadata_path = path or metadata_path_for(dataset_name)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_segment_filters_cached(
    dataset_name: str, baseline_variant: str = "real_data"
) -> dict[str, list[Any]] | None:
    """Load cached segment filter values. Returns None if cache doesn't exist."""
    cache_path = segment_filter_cache_path(dataset_name, baseline_variant)
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        # Convert lists back to sets for the return type
        return {key: list(values) for key, values in data.items()}
    except Exception as e:
        logger.warning(f"Failed to load segment filter cache from {cache_path}: {e}")
        return None


def save_segment_filters_cache(
    segment_filters: dict[str, set[Any]],
    dataset_name: str,
    baseline_variant: str,
) -> Path:
    """Save segment filter values to cache. Converts sets to sorted lists for JSON."""
    cache_path = segment_filter_cache_path(dataset_name, baseline_variant)
    _metadata_dir().mkdir(parents=True, exist_ok=True)

    # Convert sets to sorted lists for JSON serialization
    serializable = {
        key: sorted([str(v) for v in values], key=str)
        for key, values in segment_filters.items()
    }
    cache_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    logger.info(f"Saved segment filter cache to {cache_path}")
    return cache_path
