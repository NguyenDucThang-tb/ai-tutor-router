"""Dataset-backed benchmark CLI. Real mode remains explicitly gated by config."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from datasets.schemas import DatasetManifest
from datasets.validation import load_processed_dataset
from evaluation.metrics import summarize_records
from evaluation.rubrics import review_queue
from experiments.analysis import load_records
from experiments.collection_plan import validate_collection_plan
from experiments.config import load_config
from experiments.replay import replay_records, write_replay
from experiments.runner import run_benchmark, run_id_for


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run and analyze paired math benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--dataset-dir", type=Path, default=Path("data/processed/gsm8k"))
    run.add_argument("--manifest", type=Path, default=Path("data/manifests/gsm8k.json"))
    run.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    run.add_argument("--collection-plan", type=Path)
    run.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    for command in ("evaluate", "report", "review-queue"):
        item = sub.add_parser(command)
        item.add_argument("--run-dir", type=Path, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--source-records", type=Path, required=True)
    replay.add_argument("--dataset-dir", type=Path, required=True)
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--run-id", required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "run":
        config = load_config(args.config)
        source_manifest = DatasetManifest.model_validate_json(args.manifest.read_text("utf-8"))
        records = load_processed_dataset(args.dataset_dir, source_manifest)
        plan = None
        if args.collection_plan:
            plan = json.loads(args.collection_plan.read_text("utf-8"))
            selected = validate_collection_plan(
                plan,
                records,
                source_manifest,
                expected_split=args.split,
                expected_count=config.max_samples,
            )
        else:
            selected = sorted(
                (row for row in records if row.assigned_split == args.split),
                key=lambda row: row.sample_id,
            )
            attempts = config.max_retries + 1 if config.mode == "real" else 1
            if config.mode == "real" and config.fallback_policy == "operational":
                attempts += 1
            selected = selected[: min(config.max_samples, config.max_requests // (2 * attempts))]
        if not selected:
            raise ValueError("request cap leaves no paired benchmark samples")
        target = args.output_dir / run_id_for(
            config, selected, plan.get("plan_sha256") if plan is not None else None
        )
        manifest, executions = asyncio.run(
            run_benchmark(
                config,
                records,
                source_manifest,
                target,
                args.split,
                collection_plan=plan,
            )
        )
        print(
            json.dumps(
                {
                    "run_id": manifest.run_id,
                    "run_dir": str(target),
                    "records": len(executions),
                    "mode": config.mode,
                    "scientific_protocol_eligible": manifest.scientific_protocol_eligible,
                    "selection_policy": manifest.selection_policy,
                    "sample_ids": manifest.collection_plan_sample_ids
                    if plan is not None
                    else [row.sample_id for row in selected],
                    "sample_ids_sha256": manifest.sample_ids_sha256,
                    "collection_plan_sha256": manifest.collection_plan_sha256,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "replay":
        manifest = DatasetManifest.model_validate_json(args.manifest.read_text("utf-8"))
        records = load_processed_dataset(args.dataset_dir, manifest)
        replayed = replay_records(args.source_records, records, args.run_id)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_replay(args.output_dir / "records.jsonl", replayed)
        print(
            json.dumps(
                {"run_id": args.run_id, "records": len(replayed), "scientific_eligible": False},
                indent=2,
            )
        )
        return 0
    records = load_records(args.run_dir / "records.jsonl")
    result = review_queue(records) if args.command == "review-queue" else summarize_records(records)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
