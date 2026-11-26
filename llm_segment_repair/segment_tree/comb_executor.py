"""Run a comb repair sweep and deferred rebase over in-memory tables."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pandas as pd

from .builder import SegmentTreeNode
from .node_performance import _default_metric_fn, _evaluate_single_node, filter_df_for_node

SCORE = "overall_quality_score"


def run_comb_repair(
    tree: SegmentTreeNode,
    real: pd.DataFrame,
    initial: pd.DataFrame,
    metadata: dict[str, Any],
    candidates: Callable[[int, SegmentTreeNode, pd.DataFrame], pd.DataFrame],
) -> Iterator[dict[str, Any]]:
    """Yield repair and rebase decisions for schema-aligned real and synthetic tables.

    The candidate callback receives the 1-based repair index, full-tree leaf,
    and its real rows. It must return one matching candidate per real leaf row.
    """
    assert not real.empty and not initial.empty
    nodes = {node.id: node for node in tree.iter_dfs()}
    leaves = [node for node in nodes.values() if node.is_leaf]
    active = {leaf.id: filter_df_for_node(initial, leaf) for leaf in leaves}
    descendants = {
        node.id: [leaf.id for leaf in node.iter_dfs() if leaf.is_leaf]
        for node in nodes.values()
    }

    def score(node: SegmentTreeNode, replacement_id: str | None = None,
              replacement: pd.DataFrame | None = None) -> float | None:
        frames = [
            replacement if leaf_id == replacement_id else active[leaf_id]
            for leaf_id in descendants[node.id]
        ]
        if len(frames) == 1:
            frame = frames[0]
        else:
            nonempty = [part for part in frames if not part.empty]
            frame = pd.concat(nonempty, ignore_index=True) if nonempty else initial.iloc[:0]
        result = _evaluate_single_node(
            node, real_df=real, synthetic_df=frame, metadata=metadata, evaluator=_default_metric_fn,
        )
        return result.metrics.get(SCORE)

    scores = {node.id: score(node) for node in nodes.values()}
    assert scores[tree.id] is not None, "The initial table has no scorable rows in the segment tree"

    def repair_priority(node: SegmentTreeNode) -> tuple:
        value = scores[node.id]
        deficit = max(0, node.real_row_count - len(active[node.id]))
        return (float("inf") if value is None else value, node.real_row_count, deficit, node.depth, node.id)

    deferred: dict[str, pd.DataFrame] = {}
    event_index = 0
    for event_index, leaf in enumerate(sorted(leaves, key=repair_priority), start=1):
        replacement = candidates(event_index, leaf, filter_df_for_node(real, leaf))
        assert len(replacement) == leaf.real_row_count, f"Candidate row count mismatch for {leaf.id}"
        matching_rows = len(filter_df_for_node(replacement, leaf))
        assert matching_rows == len(replacement), (
            f"Candidate for {leaf.id} contains {len(replacement) - matching_rows} rows outside the leaf"
        )
        lineage = {}
        node = leaf
        while True:
            lineage[node.id] = {"before": scores[node.id], "after": score(node, leaf.id, replacement)}
            if node.parent_id is None:
                break
            node = nodes[node.parent_id]
        accepted = all(
            value["after"] is not None and (value["before"] is None or value["after"] >= value["before"])
            for value in lineage.values()
        )
        if accepted:
            active[leaf.id] = replacement
            scores.update({node_id: value["after"] for node_id, value in lineage.items()})
        else:
            deferred[leaf.id] = replacement
        yield {
            "event_index": event_index, "phase": "repair", "node_id": leaf.id,
            "accepted": int(accepted), "accepted_state_global_after": scores[tree.id],
            "lineage": lineage,
        }

    while deferred:
        ranked = [(leaf_id, frame, score(tree, leaf_id, frame)) for leaf_id, frame in deferred.items()]
        assert all(value is not None for _, _, value in ranked)
        ranked.sort(key=lambda item: item[2], reverse=True)
        accepted_this_pass = False
        for leaf_id, frame, ranked_score in ranked:
            global_score = score(tree, leaf_id, frame) if accepted_this_pass else ranked_score
            assert global_score is not None
            accepted = global_score > scores[tree.id]
            if accepted:
                active[leaf_id] = frame
                scores[tree.id] = global_score
                accepted_this_pass = True
                del deferred[leaf_id]
            event_index += 1
            yield {
                "event_index": event_index, "phase": "rebase", "node_id": leaf_id,
                "accepted": int(accepted), "accepted_state_global_after": scores[tree.id],
                "candidate_global_score": global_score,
            }
        if not accepted_this_pass:
            break
