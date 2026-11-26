"""Run one Adult Income comb experiment using local CSV inputs and checkpoints."""

import argparse
import csv
import json
import os
from hashlib import sha256
from io import BytesIO
from pathlib import Path

DATASET = "adult_income"
DEFAULT_MODEL = "gpt-5.2"
MINI_ROWS = 120
MIN_CHILD_ROWS = 30
FULL_REPLAY_MIN_CHILD_ROWS = 100
MAX_INITIAL_LLM_CALLS = 4
MAX_REPAIR_LLM_CALLS = 5
ROOT = Path(__file__).resolve().parent
CANDIDATE_MANIFEST = "manifest.json"
TRAJECTORY_COLUMNS = ("event_index", "phase", "accepted_state_global_after", "accepted")


def mini_sample(frame, seed):
    """Cap an input table at MINI_ROWS using a reproducible sample."""
    if len(frame) <= MINI_ROWS:
        return frame
    return frame.sample(n=MINI_ROWS, random_state=seed).sort_index()


def replay_inputs(real, initial, metadata, *, seed, min_child_rows):
    """Fingerprint the ordered input tables and configuration used for scoring."""
    return {
        "dataset": DATASET,
        "seed": seed,
        "min_child_rows": min_child_rows,
        "real_sha256": sha256(real.to_csv(index=False).encode("utf-8")).hexdigest(),
        "initial_sha256": sha256(initial.to_csv(index=False).encode("utf-8")).hexdigest(),
        "schema_sha256": sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--real", type=Path, help="Prepared baseline CSV.")
    baseline.add_argument("--raw", type=Path, help="Headerless UCI Adult training file.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initial", type=Path, help="Repair an existing initial synthetic CSV.")
    mode.add_argument("--generate-initial", action="store_true",
                      help="Generate the initial synthetic CSV with OpenAI.")
    parser.add_argument("--candidates", type=Path, help="Replay saved candidates and their manifest without an API key.")
    parser.add_argument("--full-replay", action="store_true",
                        help="Replay with unsampled inputs and 100-row children.")
    parser.add_argument(
        "--output", type=Path, help="Output CSV path. Defaults to a file in output/."
    )
    parser.add_argument("--run-dir", type=Path, help="New directory for candidates and decision logs.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="OpenAI model for initial generation or repair (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.generate_initial and any((args.candidates, args.full_replay, args.run_dir)):
        parser.error("--generate-initial cannot be combined with replay options or --run-dir")
    if args.full_replay and args.candidates is None:
        parser.error("--full-replay requires --candidates")
    if args.output is None:
        filename = "trajectory.csv"
        if args.generate_initial:
            filename = "adult_income_initial.csv"
        elif args.candidates is not None:
            filename = "replay.csv"
        args.output = Path("output") / filename
    if args.output.exists():
        parser.error("--output already exists")
    import pandas as pd
    from dotenv import load_dotenv

    from llm_segment_repair.metadata import load_metadata_dict
    from llm_segment_repair.segment_tree import SegmentTreeConfig, build_segment_tree
    from llm_segment_repair.segment_tree.comb_executor import run_comb_repair

    metadata = load_metadata_dict(DATASET)
    columns = list(metadata["columns"])
    if args.raw:
        from llm_segment_repair.adult_income import prepare_adult_income

        real = prepare_adult_income(args.raw).dropna()[columns]
    else:
        real = pd.read_csv(args.real, usecols=columns, float_precision="round_trip")[columns]
    if real.empty or real.isna().any().any():
        parser.error("Real CSV must contain complete Adult Income rows")
    if not args.full_replay:
        real = mini_sample(real, args.seed)

    generator = None
    if args.candidates is None:
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("OPENAI_API_KEY is required for a live run")
        max_calls = MAX_INITIAL_LLM_CALLS if args.generate_initial else MAX_REPAIR_LLM_CALLS
        generator = build_generator(metadata, args.model, args.seed, max_calls=max_calls)

    if args.generate_initial:
        initial = generator.generate(real, lambda *_: None)
        assert len(initial) == len(real), "Initial generation did not return one row per input row"
        if initial[columns].isna().any().any():
            parser.error("OpenAI returned incomplete rows for the initial CSV")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="") as output:
            initial[columns].to_csv(output, index=False)
        return

    run_dir = args.run_dir or args.output.with_suffix(".run")
    if run_dir.resolve() == args.output.resolve():
        parser.error("--run-dir and --output must be different paths")
    if run_dir.exists():
        parser.error("--run-dir already exists")

    initial = pd.read_csv(args.initial, usecols=columns, float_precision="round_trip")[columns]
    if initial.empty or initial.isna().any().any():
        parser.error("Initial CSV must contain complete Adult Income rows")
    if not args.full_replay:
        initial = mini_sample(initial, args.seed)

    min_child_rows = FULL_REPLAY_MIN_CHILD_ROWS if args.full_replay else MIN_CHILD_ROWS
    inputs = replay_inputs(real, initial, metadata, seed=args.seed, min_child_rows=min_child_rows)
    tree = build_segment_tree(real, metadata, SegmentTreeConfig(min_child_rows=min_child_rows))
    recorded_files = {}
    if args.candidates is not None:
        try:
            manifest = json.loads((args.candidates / CANDIDATE_MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot read candidate manifest: {exc}")
        if not isinstance(manifest, dict) or manifest.get("inputs") != inputs:
            parser.error("Candidate manifest does not match the input tables, schema, seed, or tree settings")
        recorded_files = manifest.get("files")
        leaf_count = sum(node.is_leaf for node in tree.iter_dfs())
        if not isinstance(recorded_files, dict) or len(recorded_files) != leaf_count:
            parser.error("Candidate manifest must contain one file per leaf")
    run_dir.mkdir(parents=True)
    candidate_dir = run_dir / "candidates"
    candidate_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "dataset": DATASET, "model": args.model if generator is not None else None,
        "seed": args.seed, "min_child_rows": min_child_rows,
        "real_rows": len(real), "initial_rows": len(initial),
        "max_llm_calls": MAX_REPAIR_LLM_CALLS if generator is not None else 0,
    }, indent=2) + "\n", encoding="utf-8")

    saved_files = {}

    def candidate(index, leaf, target):
        filename = f"{index:04d}.csv"
        if generator is None:
            entry = recorded_files.get(filename)
            if not isinstance(entry, dict) or entry.get("node_id") != leaf.id:
                parser.error(f"Candidate {filename} is not recorded for leaf {leaf.id}")
            try:
                data = (args.candidates / filename).read_bytes()
            except OSError as exc:
                parser.error(f"Cannot read candidate {filename}: {exc}")
            if sha256(data).hexdigest() != entry.get("sha256"):
                parser.error(f"Candidate {filename} differs from its manifest")
        else:
            checkpoint_path = run_dir / f"{index:04d}-iterations.jsonl"
            with checkpoint_path.open("x", encoding="utf-8") as checkpoint_log:
                def checkpoint(iteration, rows, score):
                    checkpoint_file = run_dir / f"{index:04d}-iteration-{iteration:03d}.csv"
                    rows.to_csv(checkpoint_file, index=False)
                    checkpoint_log.write(json.dumps({
                        "iteration": iteration, "node_id": leaf.id, "rows": len(rows), "score": score,
                    }) + "\n")
                    checkpoint_log.flush()

                rows = generator.generate(target, checkpoint)
            data = rows.to_csv(index=False).encode("utf-8")
        (candidate_dir / filename).write_bytes(data)
        saved_files[filename] = {"node_id": leaf.id, "sha256": sha256(data).hexdigest()}
        return pd.read_csv(BytesIO(data), usecols=columns, float_precision="round_trip")[columns]

    with (run_dir / "decisions.jsonl").open("x", encoding="utf-8") as decisions:
        trajectory = []
        for event in run_comb_repair(tree, real, initial, metadata, candidate):
            decisions.write(json.dumps(event, allow_nan=False) + "\n")
            decisions.flush()
            trajectory.append({key: event[key] for key in TRAJECTORY_COLUMNS})
    (candidate_dir / CANDIDATE_MANIFEST).write_text(
        json.dumps({"inputs": inputs, "files": saved_files}, indent=2) + "\n", encoding="utf-8"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=TRAJECTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(trajectory)


def build_generator(metadata, model, seed, *, max_calls):
    """Build the paper's iterative generation method from Adult Income metadata."""
    from llm_segment_repair.generation import (
        IterativeGenerationConfig,
        IterativeLLMDatasetSegmentGenerationGenerator,
    )

    config = IterativeGenerationConfig(
        dataset_name=DATASET, dataset_metadata=metadata, baseline_variant="real_data",
        synthetic_variant="iterative-llm", model=model,
        temperature=0.7 if model == DEFAULT_MODEL else None,
        min_iterations=1, max_additional_iterations=2, convergence_threshold=0.05,
        correlation_weight=0.3, min_overall_score_delta=0.01, max_frobenius_increase=0.1,
        refinement_max_replace_fraction=0.5, refinement_min_improvement=0.01,
        repair_fraction=0.3, repair_min_rows_per_iteration=10, rows_per_iteration=50,
        top_n_gaps=5, top_category_values=10, sample_rows=5, n_rows_multiplier=1.0,
        sample_data_csv=ROOT / "data" / "adult_income_texture.csv",
        seed=seed, disable_tier_allocation=True,
    )
    return IterativeLLMDatasetSegmentGenerationGenerator(config, max_calls=max_calls)


if __name__ == "__main__":
    main()
