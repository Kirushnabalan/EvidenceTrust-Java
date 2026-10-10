import csv
import json
from collections import Counter, deque
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "megavul_manifest.csv"
)

MEGAVUL_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "megavul"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "structural"
    / "audit"
)

DETAIL_CSV = OUTPUT_DIR / "graph_audit.csv"
SUMMARY_JSON = OUTPUT_DIR / "graph_audit_summary.json"


def resolve_graph_path(raw_path: str):
    if not raw_path:
        return None

    p = Path(raw_path.strip())

    candidates = [
        p,
        PROJECT_ROOT / p,
        MEGAVUL_ROOT / p,
        MEGAVUL_ROOT / "megavul_graph" / p,
        MEGAVUL_ROOT / "megavul_graph" / p.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


def count_components(node_ids, edges):
    adjacency = {
        node_id: set()
        for node_id in node_ids
    }

    for edge in edges:
        src = edge.get("outNode")
        dst = edge.get("inNode")

        if src in adjacency and dst in adjacency:
            adjacency[src].add(dst)
            adjacency[dst].add(src)

    visited = set()
    components = 0

    for node_id in node_ids:
        if node_id in visited:
            continue

        components += 1
        queue = deque([node_id])
        visited.add(node_id)

        while queue:
            current = queue.popleft()

            for neighbour in adjacency[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)

    isolated = sum(
        1
        for node_id in node_ids
        if not adjacency[node_id]
    )

    return components, isolated


def audit_graph(graph):
    if not isinstance(graph, dict):
        raise ValueError(
            "Graph top level is not a dictionary."
        )

    nodes = graph.get("nodes")
    edges = graph.get("edges")

    if not isinstance(nodes, list):
        raise ValueError(
            "'nodes' is not a list."
        )

    if not isinstance(edges, list):
        raise ValueError(
            "'edges' is not a list."
        )

    node_labels = Counter()
    edge_types = Counter()

    node_ids = []
    nodes_with_code = 0
    nodes_with_line = 0
    code_nodes_with_line = 0

    for node in nodes:
        if not isinstance(node, dict):
            continue

        label = node.get("_label")

        if label is not None:
            node_labels[str(label)] += 1

        node_id = node.get("id")

        if node_id is not None:
            node_ids.append(node_id)

        code = node.get("code")

        if isinstance(code, str) and code.strip():
            nodes_with_code += 1

            if node.get("lineNumber") is not None:
                code_nodes_with_line += 1

        if node.get("lineNumber") is not None:
            nodes_with_line += 1

    node_id_set = set(node_ids)

    duplicate_node_ids = (
        len(node_ids) - len(node_id_set)
    )

    invalid_edge_endpoints = 0

    for edge in edges:
        if not isinstance(edge, dict):
            invalid_edge_endpoints += 1
            continue

        etype = edge.get("etype")

        if etype is not None:
            edge_types[str(etype)] += 1

        src = edge.get("outNode")
        dst = edge.get("inNode")

        if (
            src not in node_id_set
            or dst not in node_id_set
        ):
            invalid_edge_endpoints += 1

    component_count, isolated_count = (
        count_components(
            node_id_set,
            edges,
        )
    )

    node_count = len(nodes)
    edge_count = len(edges)

    return {
        "node_count": node_count,
        "edge_count": edge_count,

        "unique_node_ids": len(node_id_set),
        "duplicate_node_ids": duplicate_node_ids,

        "nodes_with_code": nodes_with_code,
        "nodes_with_line_number": nodes_with_line,
        "code_nodes_with_line_number":
            code_nodes_with_line,

        "line_number_ratio_all_nodes":
            (
                nodes_with_line / node_count
                if node_count
                else 0.0
            ),

        "line_number_ratio_code_nodes":
            (
                code_nodes_with_line
                / nodes_with_code
                if nodes_with_code
                else 0.0
            ),

        "invalid_edge_endpoints":
            invalid_edge_endpoints,

        "component_count":
            component_count,

        "isolated_node_count":
            isolated_count,

        "isolated_node_ratio":
            (
                isolated_count / node_count
                if node_count
                else 0.0
            ),

        "node_labels": node_labels,
        "edge_types": edge_types,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        manifest = list(csv.DictReader(f))

    # Development data only.
    rows = [
        row
        for row in manifest
        if row.get("split")
        in {"train", "validation"}
    ]

    print(
        "Auditing development graphs:",
        f"{len(rows):,}",
    )

    detailed_rows = []

    global_node_labels = Counter()
    global_edge_types = Counter()

    missing_paths = 0
    load_failures = 0
    malformed_graphs = 0

    empty_graphs = 0
    graphs_with_invalid_endpoints = 0
    graphs_with_duplicate_ids = 0

    for index, row in enumerate(
        rows,
        start=1,
    ):
        result = {
            "sample_id": row.get("sample_id"),
            "split": row.get("split"),
            "is_vul": row.get("is_vul"),
            "graph_path": row.get("graph_path"),
            "resolved": 0,
            "load_success": 0,
            "valid_structure": 0,
            "error": "",
        }

        graph_path = resolve_graph_path(
            row.get("graph_path", "")
        )

        if graph_path is None:
            missing_paths += 1

            result["error"] = (
                "graph_path_not_resolved"
            )

            detailed_rows.append(result)
            continue

        result["resolved"] = 1

        try:
            with graph_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                graph = json.load(f)

            result["load_success"] = 1

        except Exception as exc:
            load_failures += 1

            result["error"] = (
                f"load_failed:"
                f"{type(exc).__name__}"
            )

            detailed_rows.append(result)
            continue

        try:
            audit = audit_graph(graph)

            result["valid_structure"] = 1

        except Exception as exc:
            malformed_graphs += 1

            result["error"] = (
                f"invalid_structure:"
                f"{type(exc).__name__}"
            )

            detailed_rows.append(result)
            continue

        for key in [
            "node_count",
            "edge_count",
            "unique_node_ids",
            "duplicate_node_ids",
            "nodes_with_code",
            "nodes_with_line_number",
            "code_nodes_with_line_number",
            "line_number_ratio_all_nodes",
            "line_number_ratio_code_nodes",
            "invalid_edge_endpoints",
            "component_count",
            "isolated_node_count",
            "isolated_node_ratio",
        ]:
            result[key] = audit[key]

        if (
            audit["node_count"] == 0
            or audit["edge_count"] == 0
        ):
            empty_graphs += 1

        if audit["invalid_edge_endpoints"] > 0:
            graphs_with_invalid_endpoints += 1

        if audit["duplicate_node_ids"] > 0:
            graphs_with_duplicate_ids += 1

        global_node_labels.update(
            audit["node_labels"]
        )

        global_edge_types.update(
            audit["edge_types"]
        )

        result["node_label_types"] = "|".join(
            sorted(audit["node_labels"])
        )

        result["edge_types"] = "|".join(
            sorted(audit["edge_types"])
        )

        detailed_rows.append(result)

        if index % 1000 == 0:
            print(
                f"Processed "
                f"{index:,}/{len(rows):,}"
            )

    fieldnames = sorted(
        {
            key
            for row in detailed_rows
            for key in row.keys()
        }
    )

    with DETAIL_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(detailed_rows)

    successful = [
        row
        for row in detailed_rows
        if row.get("valid_structure") == 1
    ]

    summary = {
        "scope": "train+validation only",

        "samples_total":
            len(rows),

        "resolved_graphs":
            sum(
                int(row.get("resolved", 0))
                for row in detailed_rows
            ),

        "loaded_graphs":
            sum(
                int(row.get("load_success", 0))
                for row in detailed_rows
            ),

        "valid_graphs":
            len(successful),

        "missing_paths":
            missing_paths,

        "load_failures":
            load_failures,

        "malformed_graphs":
            malformed_graphs,

        "empty_graphs":
            empty_graphs,

        "graphs_with_invalid_edge_endpoints":
            graphs_with_invalid_endpoints,

        "graphs_with_duplicate_node_ids":
            graphs_with_duplicate_ids,

        "node_labels":
            dict(
                global_node_labels.most_common()
            ),

        "edge_types":
            dict(
                global_edge_types.most_common()
            ),
    }

    if successful:
        for metric in [
            "node_count",
            "edge_count",
            "line_number_ratio_all_nodes",
            "line_number_ratio_code_nodes",
            "component_count",
            "isolated_node_ratio",
        ]:
            values = [
                float(row[metric])
                for row in successful
                if row.get(metric) not in (
                    None,
                    "",
                )
            ]

            if values:
                summary[
                    f"{metric}_mean"
                ] = (
                    sum(values) / len(values)
                )

                summary[
                    f"{metric}_min"
                ] = min(values)

                summary[
                    f"{metric}_max"
                ] = max(values)

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("STRUCTURAL AUDIT COMPLETE")
    print("=" * 80)

    print(
        "Samples:",
        summary["samples_total"],
    )

    print(
        "Resolved:",
        summary["resolved_graphs"],
    )

    print(
        "Valid:",
        summary["valid_graphs"],
    )

    print(
        "Missing paths:",
        missing_paths,
    )

    print(
        "Load failures:",
        load_failures,
    )

    print(
        "Malformed:",
        malformed_graphs,
    )

    print(
        "Empty graphs:",
        empty_graphs,
    )

    print(
        "Invalid endpoint graphs:",
        graphs_with_invalid_endpoints,
    )

    print()
    print("NODE LABELS")
    print("-" * 80)

    for label, count in (
        global_node_labels.most_common()
    ):
        print(
            f"{label:30s} {count:,}"
        )

    print()
    print("EDGE TYPES")
    print("-" * 80)

    for edge_type, count in (
        global_edge_types.most_common()
    ):
        print(
            f"{edge_type:30s} {count:,}"
        )

    print()
    print("Saved:")
    print(DETAIL_CSV)
    print(SUMMARY_JSON)


if __name__ == "__main__":
    main()