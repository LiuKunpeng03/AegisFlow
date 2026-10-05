#!/usr/bin/env python3

"""Aggregate CLIP, LPIPS, and ArcFace similarity from metrics logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_FIELDS = ("clip_score", "lpips_score", "ArcFace_similarity", "time", "peak_memory")
DEFAULT_METHOD = "aegisflow"


def iter_metric_files(root: Path, method: str | None = None) -> Iterator[Path]:
    for metrics_file in sorted(root.rglob("metrics.json")):
        if metrics_file.is_file():
            if method is not None and len(metrics_file.parts) >= 3:
                if metrics_file.parts[-2] != method:
                    continue
            yield metrics_file


def iter_metric_rows(metrics_file: Path) -> Iterator[dict]:
    with metrics_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, dict):
                yield value


def collect_values(
    root: Path, fields: Iterable[str], method: str | None = None
) -> dict[str, list[float]]:
    values = {field: [] for field in fields}

    for metrics_file in iter_metric_files(root, method=method):
        try:
            for row in iter_metric_rows(metrics_file):
                for field in fields:
                    value = row.get(field)
                    if isinstance(value, (int, float)):
                        values[field].append(float(value))
        except Exception as exc:
            print(f"Skipping {metrics_file}: {exc}")

    return values


def summarize(values: dict[str, list[float]]) -> dict[str, dict[str, float | int | None]]:
    summary = {}
    for field, items in values.items():
        if items:
            summary[field] = {
                "count": len(items),
                "mean": sum(items) / len(items),
            }
        else:
            summary[field] = {"count": 0, "mean": None}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average CLIP, LPIPS, and ArcFace similarity from metrics.json logs."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="output",
        help="Root directory containing metrics.json files (default: output)",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=list(DEFAULT_FIELDS),
        help="Metric fields to average (default: clip_score lpips_score ArcFace_similarity)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=DEFAULT_METHOD,
        help="Only include metrics under this subdirectory name (default: oc_flow)",
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Include metrics from every method subdirectory instead of filtering by --method.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    method = None if args.all_methods else args.method
    values = collect_values(root, args.fields, method=method)
    summary = summarize(values)

    print(f"Scanned: {root.resolve()}")
    if method is not None:
        print(f"Method filter: {method}")
    for field in args.fields:
        field_summary = summary[field]
        mean_value = field_summary["mean"]
        if mean_value is None:
            print(f"{field}: no values found")
        else:
            print(f"{field}: {mean_value:.6f} (n={field_summary['count']})")


if __name__ == "__main__":
    main()