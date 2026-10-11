
from __future__ import annotations

import argparse
import csv
import importlib
import shutil

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

from .source_selection import select_evidence
from .graph_paths import check_graph_reference
from .graph_evidence import summarize_graph_file


FIELDS = [
    "record_index",
    "label",
    "graph_reference",
    "status",
    "node_count",
    "edge_count",
    "ast_edges",
    "cfg_edges",
    "reaching_def_edges",
    "cdg_edges",
    "call_nodes",
    "invalid_nodes",
    "duplicate_node_ids",
    "invalid_edges",
    "missing_edge_endpoints",
    "error",
]


def batch_path(output_root: Path, start: int) -> Path:
    return output_root / f"graph_batch_{start:06d}.csv"


def find_resume_index(
    output_root: Path,
    batch_size: int,
    total_records: int,
) -> int:
    start = 0

    while start < total_records:
        path = batch_path(output_root, start)

        if not path.is_file():
            break

        expected = min(batch_size, total_records - start)

        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames != FIELDS:
                raise ValueError(f"Invalid batch columns: {path}")

            count = 0

            for row in reader:
                if row["record_index"] != str(start + count):
                    raise ValueError(f"Invalid record index: {path}")

                count += 1

        if count != expected:
            raise ValueError(f"Incomplete batch: {path}")

        start += count

    return start


def inspect_reference(graph_root: Path, reference: str):
    try:
        result = check_graph_reference(graph_root, reference)

        if not result.exists:
            return "missing", {}, "Graph file not found"

        summary = summarize_graph_file(result.resolved_path)

        return "ok", asdict(summary), ""

    except (ValueError, OSError, TypeError) as error:
        return "error", {}, str(error)


def run_audit(
    dataset_path: Path,
    graph_root: Path,
    output_root: Path,
    batch_size: int = 500,
    max_batches: int = 5,
    workers: int = 4,
    total_records: int = 41949,
):
    import ijson

    if batch_size <= 0 or max_batches <= 0 or workers <= 0:
        raise ValueError("Batch size, max batches and workers must be positive")

    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    if not graph_root.is_dir():
        raise NotADirectoryError(graph_root)

    output_root.mkdir(parents=True, exist_ok=True)

    start = find_resume_index(
        output_root,
        batch_size,
        total_records,
    )

    print("EVIDENCETRUST-JAVA | GRAPH AUDIT")
    print(f"Resuming from record: {start:,}")

    if start >= total_records:
        print("All batches completed")
        return

    end = min(total_records, start + batch_size * max_batches)

    cache = {}
    run_counts = Counter()

    with dataset_path.open("rb") as file:
        records = ijson.items(file, "item")

        for _ in islice(records, start):
            pass

        with (
            ThreadPoolExecutor(max_workers=workers) as executor,
            TemporaryDirectory() as temp_dir,
        ):
            for batch_start in range(start, end, batch_size):
                batch = list(islice(records, batch_size))

                expected = min(batch_size, total_records - batch_start)

                if len(batch) != expected:
                    raise RuntimeError("Dataset ended unexpectedly")

                selected = [
                    (batch_start + index, select_evidence(record))
                    for index, record in enumerate(batch)
                ]

                references = list(dict.fromkeys(
                    evidence.graph_reference
                    for _, evidence in selected
                    if evidence.graph_reference not in cache
                ))

                results = executor.map(
                    lambda reference: inspect_reference(
                        graph_root, reference
                    ),
                    references,
                )

                cache.update(zip(references, results))

                target = batch_path(output_root, batch_start)
                local_path = Path(temp_dir) / target.name

                batch_counts = Counter()

                with local_path.open(
                    "w", newline="", encoding="utf-8"
                ) as output:
                    writer = csv.DictWriter(output, fieldnames=FIELDS)
                    writer.writeheader()

                    for index, evidence in selected:
                        status, metrics, error = cache[
                            evidence.graph_reference
                        ]

                        writer.writerow({
                            "record_index": index,
                            "label": evidence.label,
                            "graph_reference": evidence.graph_reference,
                            "status": status,
                            "error": error,
                            **metrics,
                        })

                        batch_counts[status] += 1

                temporary_target = output_root / f".{target.name}.tmp"

                shutil.copyfile(local_path, temporary_target)
                temporary_target.replace(target)

                run_counts.update(batch_counts)

                print(
                    f"Records {batch_start:,}-"
                    f"{batch_start + len(batch) - 1:,} | "
                    f"OK: {batch_counts['ok']} | "
                    f"Missing: {batch_counts['missing']} | "
                    f"Errors: {batch_counts['error']} | SAVED"
                )

    print("\nRUN SUMMARY")
    print(f"Processed: {sum(run_counts.values()):,}")
    print(f"Successful: {run_counts['ok']:,}")
    print(f"Missing: {run_counts['missing']:,}")
    print(f"Errors: {run_counts['error']:,}")
    print(
        f"Next record: "
        f"{find_resume_index(output_root, batch_size, total_records):,}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Resumable MegaVul-Java graph-content audit"
    )

    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-batches", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--total-records", type=int, default=41949)

    args = parser.parse_args()

    run_audit(
        dataset_path=args.dataset,
        graph_root=args.graph_root,
        output_root=args.output,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        workers=args.workers,
        total_records=args.total_records,
    )


if __name__ == "__main__":
    main()
