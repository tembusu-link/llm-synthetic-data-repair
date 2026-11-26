"""Plot the root quality retained after each comb decision."""

import argparse
import csv
import json
from pathlib import Path


def read_trajectory(path):
    """Read event_index, phase, accepted_state_global_after and accepted from CSV."""
    with path.open(newline="", encoding="utf-8") as source:
        return [
            (
                int(row["event_index"]),
                row["phase"],
                float(row["accepted_state_global_after"]),
                bool(int(row["accepted"])),
            )
            for row in csv.DictReader(source)
        ]


def redraw(source, output, *, decisions=None):
    """Plot a trajectory and its initial score when the decision log is available."""
    rows = read_trajectory(source)
    if not rows:
        raise ValueError("Trajectory CSV has no events")

    if decisions is None:
        sibling = source.with_suffix(".run") / "decisions.jsonl"
        if sibling.is_file():
            decisions = sibling
    if decisions is not None:
        with decisions.open(encoding="utf-8") as log:
            first = json.loads(log.readline())
        event, phase, score, accepted = rows[0]
        if (
            first["event_index"], first["phase"],
            first["accepted_state_global_after"], bool(first["accepted"]),
        ) != (event, phase, score, accepted) or phase != "repair":
            raise ValueError("Decision log does not match the first repair in the trajectory")
        rows.insert(0, (0, "initial", first["lineage"]["root"]["before"], False))

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    events, phases, scores, accepted = zip(*rows, strict=True)
    (line,) = axis.plot(events, scores)
    color = line.get_color()
    for label, marker, face, indices in (
        ("Initial score", "s", color, [i for i, phase in enumerate(phases) if phase == "initial"]),
        ("Accepted", "o", color, [i for i, flag in enumerate(accepted) if flag]),
        ("Rejected", "o", "white", [
            i for i, flag in enumerate(accepted) if not flag and phases[i] != "initial"
        ]),
    ):
        if indices:
            axis.scatter(
                [events[i] for i in indices], [scores[i] for i in indices],
                marker=marker, facecolors=face, edgecolors=color, s=28, label=label, zorder=3,
            )

    if "rebase" in phases:
        axis.axvline(events[phases.index("rebase")] - 0.5, color="gray", linestyle="--", alpha=0.6)
    if len(rows) <= 8:
        axis.set_xticks(events, [
            "Initial" if phase == "initial" else f"{phase.title()} {event}"
            for event, phase in zip(events, phases, strict=True)
        ])
        for event, score in zip(events, scores, strict=True):
            axis.annotate(
                f"{score:.4f}", (event, score), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9,
            )
        axis.margins(x=0.1, y=0.22)
    else:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlabel("Decision number")
    axis.set(title="Adult Income comb repair", ylabel="Root quality score (SDMetrics)")
    axis.legend(loc="upper left")
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Trajectory CSV from an Adult Income run.")
    parser.add_argument(
        "--decisions", type=Path, help="Decision log for a custom run directory.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/trajectory.pdf"),
        help="Figure path. Defaults to output/trajectory.pdf.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        redraw(args.input, args.output, decisions=args.decisions)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
