import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.structural.graph_parser import (
    load_graph,
)


MANIFEST_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "megavul_manifest.csv"
)

GRAPH_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "megavul"
    / "megavul_graph"
)


def main():
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    row = next(
        row
        for row in rows
        if (
            row.get("split")
            in {"train", "validation"}
            and row.get("graph_path")
        )
    )

    graph_path = (
        GRAPH_ROOT
        / row["graph_path"]
    )

    graph = load_graph(
        graph_path
    )

    print(
        "sample_id:",
        row["sample_id"],
    )

    print(
        "split:",
        row["split"],
    )

    print(
        "label:",
        row["is_vul"],
    )

    print(
        "nodes:",
        graph.num_nodes,
    )

    print(
        "edges:",
        graph.num_edges,
    )

    print()
    print("NODE LABEL COUNTS")

    for label, count in (
        graph.node_label_counts.most_common()
    ):
        print(
            f"{label:30s} {count}"
        )

    print()
    print("EDGE TYPE COUNTS")

    for edge_type, count in (
        graph.edge_type_counts.most_common()
    ):
        print(
            f"{edge_type:30s} {count}"
        )

    print()
    print("FIRST 5 NODES")

    for node in graph.nodes[:5]:
        print(
            {
                "index": node.index,
                "joern_id": node.joern_id,
                "label": node.label,
                "line": node.line_number,
                "code": node.code,
            }
        )

    print()
    print("FIRST 5 EDGES")

    for edge in graph.edges[:5]:
        print(
            {
                "source": edge.source,
                "target": edge.target,
                "type": edge.edge_type,
            }
        )

    print()
    print(
        "GRAPH PARSER SMOKE TEST PASSED"
    )


if __name__ == "__main__":
    main()