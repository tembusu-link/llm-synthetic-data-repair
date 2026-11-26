"""Generate one segment in batches using iterative SDMetrics feedback."""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd

from .gaps import _ensure_min_rows
from .iteration_base import IterativeLLMDatasetSegmentGeneratorBase


class IterativeLLMDatasetSegmentGenerationGenerator(IterativeLLMDatasetSegmentGeneratorBase):
    """Run the iterative LLM generator for a fresh segment."""

    def generate(
        self,
        target_df: pd.DataFrame,
        checkpoint: Callable[[int, pd.DataFrame, float], None],
    ) -> pd.DataFrame:
        """Generate one replacement row per target row, checkpointing every batch."""
        from llm_segment_repair.metrics import evaluate_datasets

        target_rows = len(target_df)
        assert target_rows > 0 and set(self.schema.columns) <= set(target_df)
        _, rows_model = self._get_structured_models(self.schema)
        batch_size = self.config.rows_per_iteration
        if self.config.min_iterations > 1 and target_rows >= self.config.min_iterations:
            batch_size = min(batch_size, target_rows // self.config.min_iterations)
        max_iterations = max(
            self.config.max_additional_iterations, math.ceil(target_rows / batch_size)
        )
        rows = pd.DataFrame()
        report = None
        for iteration in range(1, max_iterations + 1):
            remaining = target_rows - len(rows)
            if remaining <= 0:
                break
            requested = min(remaining, batch_size)
            if 0 < remaining - requested < 10 and requested >= batch_size:
                requested = remaining
            payload = self._build_iterative_prompt_payload(
                target_df=target_df,
                current_synthetic_df=None if rows.empty else rows,
                schema=self.schema,
                n_rows=max(self.config.prompt_row_floor, requested),
                iteration_num=iteration,
                target_total_rows=target_rows,
                total_iterations=max_iterations,
                current_report=report,
            )
            for key in ("_gap_directives_raw", "_debug_gap_directives_raw", "_target_texture_profiles"):
                payload.pop(key, None)
            batch = self._invoke_structured_rows(
                payload=payload,
                rows_model=rows_model,
                expected_columns=self.schema.columns,
                expected_n_rows=requested,
                batch_buffer_factor=self.config.batch_buffer_factor,
                prompt_row_floor=self.config.prompt_row_floor,
            )
            assert not batch.empty, f"Model returned no rows at iteration {iteration}"
            rows = pd.concat([rows, batch], ignore_index=True)
            report = evaluate_datasets(
                real_df=_ensure_min_rows(target_df),
                syn_df=_ensure_min_rows(rows),
                schema=self.schema,
            )
            checkpoint(iteration, rows, report["overall_quality_score"])
        assert len(rows) == target_rows, f"Generated {len(rows)} of {target_rows} required rows"
        return rows
