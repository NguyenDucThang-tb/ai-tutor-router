from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.schemas import DatasetManifest
from datasets.validation import load_processed_dataset
from evaluation.metrics import summarize_records
from evaluation.rubrics import phase3a_review_sample, review_queue
from experiments.records import ExperimentRecord
from experiments.replay import replay_records, write_replay


def load_records(path: Path) -> list[ExperimentRecord]:
    return [
        ExperimentRecord.model_validate_json(line) for line in path.read_text("utf-8").splitlines()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize, replay, or prepare review candidates")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("summarize", "review-queue", "review-sample"):
        command_parser = sub.add_parser(command)
        command_parser.add_argument("--records", type=Path, required=True)
        command_parser.add_argument("--output", type=Path)
        if command == "review-sample":
            command_parser.add_argument("--per-status", type=int, default=5)
            command_parser.add_argument("--seed", type=int, default=42)
    replay = sub.add_parser("replay")
    replay.add_argument("--records", type=Path, required=True)
    replay.add_argument("--dataset-dir", type=Path, required=True)
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--run-id", required=True)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "summarize":
        result = summarize_records(load_records(args.records))
    elif args.command == "review-queue":
        result = review_queue(load_records(args.records))
    elif args.command == "review-sample":
        result = phase3a_review_sample(
            load_records(args.records), per_status=args.per_status, seed=args.seed
        )
    else:
        manifest = DatasetManifest.model_validate_json(args.manifest.read_text("utf-8"))
        records = load_processed_dataset(args.dataset_dir, manifest)
        replayed = replay_records(args.records, records, args.run_id)
        write_replay(args.output, replayed)
        result = {
            "run_id": args.run_id,
            "records": len(replayed),
            "mode": "replay",
            "scientific_eligible": False,
        }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if getattr(args, "output", None) and args.command != "replay":
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
