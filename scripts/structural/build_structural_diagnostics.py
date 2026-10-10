from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.structural.graph_parser import load_graph


MODEL_INPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "model_inputs"
)

GRAPH_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "megavul"
    / "megavul_graph"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "structural"
    / "diagnostics"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "structural_diagnostics_development.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "structural_diagnostics_summary.json"
)


# Relations observed in the full Joern audit.
#
# IMPORTANT:
# Presence/absence is recorded as an observable property.
# Absence is NOT automatically interpreted as representation failure.
RELATIONS = [
    "AST",
    "CFG",
    "CDG",
    "REACHING_DEF",
    "CALL",
    "REF",
]


# ------------------------------------------------------------
# Input helpers
# ------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object in {path} "
                    f"at line {line_number}"
                )

            rows.append(row)

    return rows


def normalize_binary_label(value: Any) -> int:
    """
    Convert the model-input label to 0/1 safely.
    """

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        if value in (0, 1):
            return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "1",
            "true",
            "yes",
        }:
            return 1

        if normalized in {
            "0",
            "false",
            "no",
        }:
            return 0

    raise ValueError(
        f"Unsupported binary label: {value!r}"
    )


# ------------------------------------------------------------
# Source representation helpers
# ------------------------------------------------------------

def source_line_information(
    code: str,
) -> tuple[int, set[int]]:
    """
    Return:

    1. total number of source lines
    2. 1-based indices of non-blank source lines

    The resulting coverage measure represents observable
    source-to-graph line mapping only.

    It is NOT:
    - graph correctness
    - vulnerability-evidence sufficiency
    - prediction correctness
    - a direct adequacy score
    """

    lines = code.splitlines()

    total_lines = len(lines)

    nonblank_lines = {
        index
        for index, line in enumerate(
            lines,
            start=1,
        )
        if line.strip()
    }

    return (
        total_lines,
        nonblank_lines,
    )


# ------------------------------------------------------------
# Per-sample diagnostics
# ------------------------------------------------------------

def diagnostic_for_sample(
    row: dict[str, Any],
) -> dict[str, Any]:

    sample_id = row.get("sample_id")
    split = row.get("split")

    result: dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "is_vul": "",
        "graph_path": row.get(
            "graph_path",
            "",
        ),
        "graph_parse_success": 0,
        "diagnostic_error": "",
    }

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    try:
        result["is_vul"] = normalize_binary_label(
            row.get("is_vul")
        )

    except Exception as exc:
        result["diagnostic_error"] = (
            f"invalid_label:"
            f"{type(exc).__name__}: {exc}"
        )

        return result

    # --------------------------------------------------------
    # Required inputs
    # --------------------------------------------------------

    code = row.get("code")

    if code is None:
        code = ""

    if not isinstance(code, str):
        code = str(code)

    graph_path_value = row.get("graph_path")

    if not graph_path_value:
        result["diagnostic_error"] = (
            "missing_graph_path"
        )

        return result

    graph_path = (
        GRAPH_ROOT
        / str(graph_path_value)
    )

    # --------------------------------------------------------
    # Graph parsing
    # --------------------------------------------------------

    try:
        graph = load_graph(
            graph_path
        )

    except Exception as exc:
        result["diagnostic_error"] = (
            f"graph_parse_failed:"
            f"{type(exc).__name__}: {exc}"
        )

        return result

    result["graph_parse_success"] = 1

    # --------------------------------------------------------
    # Basic graph observations
    # --------------------------------------------------------

    result["node_count"] = graph.num_nodes
    result["edge_count"] = graph.num_edges

    # --------------------------------------------------------
    # Source-grounded line mapping
    # --------------------------------------------------------

    (
        total_source_lines,
        nonblank_source_lines,
    ) = source_line_information(
        code
    )

    graph_line_numbers = {
        node.line_number
        for node in graph.nodes
        if (
            node.line_number is not None
            and node.line_number > 0
        )
    }

    in_range_graph_lines = {
        line
        for line in graph_line_numbers
        if (
            1
            <= line
            <= total_source_lines
        )
    }

    out_of_range_graph_lines = {
        line
        for line in graph_line_numbers
        if (
            line < 1
            or line > total_source_lines
        )
    }

    mapped_nonblank_lines = (
        nonblank_source_lines
        & graph_line_numbers
    )

    result[
        "source_total_lines"
    ] = total_source_lines

    result[
        "source_nonblank_lines"
    ] = len(
        nonblank_source_lines
    )

    result[
        "graph_distinct_line_numbers"
    ] = len(
        graph_line_numbers
    )

    result[
        "graph_in_range_line_numbers"
    ] = len(
        in_range_graph_lines
    )

    result[
        "graph_out_of_range_line_numbers"
    ] = len(
        out_of_range_graph_lines
    )

    result[
        "mapped_nonblank_source_lines"
    ] = len(
        mapped_nonblank_lines
    )

    # Source-line mapping is only applicable
    # when there is at least one non-blank source line.
    mapping_applicable = (
        len(nonblank_source_lines) > 0
    )

    result[
        "source_line_mapping_applicable"
    ] = int(mapping_applicable)

    if mapping_applicable:
        result[
            "source_line_mapping_coverage"
        ] = (
            len(mapped_nonblank_lines)
            / len(nonblank_source_lines)
        )

    else:
        result[
            "source_line_mapping_coverage"
        ] = None

    # --------------------------------------------------------
    # Node-level source-location observations
    # --------------------------------------------------------

    nodes_with_line = sum(
        1
        for node in graph.nodes
        if node.line_number is not None
    )

    nodes_with_code = sum(
        1
        for node in graph.nodes
        if (
            node.code is not None
            and node.code.strip()
        )
    )

    code_nodes_with_line = sum(
        1
        for node in graph.nodes
        if (
            node.code is not None
            and node.code.strip()
            and node.line_number is not None
        )
    )

    result[
        "nodes_with_line_number"
    ] = nodes_with_line

    result[
        "node_line_number_ratio"
    ] = (
        nodes_with_line
        / graph.num_nodes
        if graph.num_nodes
        else 0.0
    )

    result[
        "nodes_with_code"
    ] = nodes_with_code

    result[
        "code_nodes_with_line_number"
    ] = code_nodes_with_line

    result[
        "code_node_line_number_ratio"
    ] = (
        code_nodes_with_line
        / nodes_with_code
        if nodes_with_code
        else 0.0
    )

    # --------------------------------------------------------
    # Relation observations
    #
    # IMPORTANT:
    # relation_present == observable presence only.
    #
    # relation_present == 0 does NOT automatically mean
    # extraction failed or that the graph is inadequate.
    #
    # Applicability rules will be analysed separately.
    # --------------------------------------------------------

    for relation in RELATIONS:
        edge_count = (
            graph.edge_type_counts.get(
                relation,
                0,
            )
        )

        prefix = relation.lower()

        result[
            f"{prefix}_edge_count"
        ] = edge_count

        result[
            f"{prefix}_present"
        ] = int(
            edge_count > 0
        )

    # --------------------------------------------------------
    # UNKNOWN node observations
    # --------------------------------------------------------

    unknown_nodes = (
        graph.node_label_counts.get(
            "UNKNOWN",
            0,
        )
    )

    result[
        "unknown_node_count"
    ] = unknown_nodes

    result[
        "unknown_node_ratio"
    ] = (
        unknown_nodes
        / graph.num_nodes
        if graph.num_nodes
        else 0.0
    )

    return result


# ------------------------------------------------------------
# Summary helpers
# ------------------------------------------------------------

def numeric_values(
    rows: list[dict[str, Any]],
    metric: str,
) -> list[float]:

    values: list[float] = []

    for row in rows:
        value = row.get(metric)

        if value is None:
            continue

        if value == "":
            continue

        try:
            values.append(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

    return values


def summarize(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:

    valid = [
        row
        for row in rows
        if row.get(
            "graph_parse_success"
        ) == 1
    ]

    summary: dict[str, Any] = {
        "scope": (
            "train+validation only"
        ),
        "samples_total": len(rows),
        "samples_valid": len(valid),
        "samples_failed": (
            len(rows)
            - len(valid)
        ),
    }

    if not valid:
        return summary

    numeric_metrics = [
        "node_count",
        "edge_count",
        "source_total_lines",
        "source_nonblank_lines",
        "graph_distinct_line_numbers",
        "graph_in_range_line_numbers",
        "graph_out_of_range_line_numbers",
        "mapped_nonblank_source_lines",
        "source_line_mapping_coverage",
        "node_line_number_ratio",
        "code_node_line_number_ratio",
        "unknown_node_count",
        "unknown_node_ratio",
    ]

    for metric in numeric_metrics:
        values = numeric_values(
            valid,
            metric,
        )

        if not values:
            continue

        summary[
            f"{metric}_mean"
        ] = (
            sum(values)
            / len(values)
        )

        summary[
            f"{metric}_min"
        ] = min(values)

        summary[
            f"{metric}_max"
        ] = max(values)

        summary[
            f"{metric}_median"
        ] = median(values)

    # --------------------------------------------------------
    # Relation presence
    # --------------------------------------------------------

    for relation in RELATIONS:
        prefix = relation.lower()

        present_key = (
            f"{prefix}_present"
        )

        present_count = sum(
            int(
                row.get(
                    present_key,
                    0,
                )
            )
            for row in valid
        )

        summary[
            f"{prefix}_present_graphs"
        ] = present_count

        summary[
            f"{prefix}_present_ratio"
        ] = (
            present_count
            / len(valid)
        )

    # --------------------------------------------------------
    # Other aggregate observations
    # --------------------------------------------------------

    summary[
        "graphs_with_unknown_nodes"
    ] = sum(
        int(
            row.get(
                "unknown_node_count",
                0,
            )
            > 0
        )
        for row in valid
    )

    summary[
        "graphs_with_out_of_range_lines"
    ] = sum(
        int(
            row.get(
                "graph_out_of_range_line_numbers",
                0,
            )
            > 0
        )
        for row in valid
    )

    summary[
        "source_line_mapping_applicable_graphs"
    ] = sum(
        int(
            row.get(
                "source_line_mapping_applicable",
                0,
            )
        )
        for row in valid
    )

    return summary


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, Any]] = []

    # Development data only.
    # Test remains locked.
    for split in (
        "train",
        "validation",
    ):
        path = (
            MODEL_INPUT_DIR
            / f"{split}.jsonl"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Model-input file not found: {path}"
            )

        split_rows = read_jsonl(
            path
        )

        print(
            f"{split}: "
            f"{len(split_rows):,}"
        )

        rows.extend(
            split_rows
        )

    print()

    print(
        "Building structural diagnostics for:",
        f"{len(rows):,}",
        "samples",
    )

    diagnostics: list[
        dict[str, Any]
    ] = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        result = diagnostic_for_sample(
            row
        )

        diagnostics.append(
            result
        )

        if index % 1000 == 0:
            print(
                f"Processed "
                f"{index:,}/"
                f"{len(rows):,}"
            )

    # --------------------------------------------------------
    # Save detailed CSV
    # --------------------------------------------------------

    fieldnames = sorted(
        {
            key
            for row in diagnostics
            for key in row.keys()
        }
    )

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            diagnostics
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = summarize(
        diagnostics
    )

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "STRUCTURAL DIAGNOSTICS COMPLETE"
    )
    print("=" * 80)

    print(
        "Samples:",
        summary["samples_total"],
    )

    print(
        "Valid:",
        summary["samples_valid"],
    )

    print(
        "Failed:",
        summary["samples_failed"],
    )

    print()

    mean_coverage = summary.get(
        "source_line_mapping_coverage_mean",
        0.0,
    )

    median_coverage = summary.get(
        "source_line_mapping_coverage_median",
        0.0,
    )

    print(
        "Mean source-line mapping coverage:",
        f"{mean_coverage:.4f}",
    )

    print(
        "Median source-line mapping coverage:",
        f"{median_coverage:.4f}",
    )

    print(
        "Graphs with out-of-range lines:",
        summary.get(
            "graphs_with_out_of_range_lines",
            0,
        ),
    )

    print(
        "Graphs with UNKNOWN nodes:",
        summary.get(
            "graphs_with_unknown_nodes",
            0,
        ),
    )

    print(
        "Mapping-applicable graphs:",
        summary.get(
            "source_line_mapping_applicable_graphs",
            0,
        ),
    )

    print()
    print("RELATION PRESENCE")
    print("-" * 80)

    for relation in RELATIONS:
        prefix = relation.lower()

        count = summary.get(
            f"{prefix}_present_graphs",
            0,
        )

        ratio = summary.get(
            f"{prefix}_present_ratio",
            0.0,
        )

        print(
            f"{relation:15s} "
            f"{count:6d} "
            f"({ratio:.2%})"
        )

    print()
    print("Saved:")
    print(OUTPUT_CSV)
    print(SUMMARY_JSON)


if __name__ == "__main__":
    main()