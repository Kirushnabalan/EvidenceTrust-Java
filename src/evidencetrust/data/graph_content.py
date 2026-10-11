
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GraphInspection:
    node_count: int
    edge_count: int
    node_labels: dict[str, int]
    edge_types: dict[str, int]
    invalid_nodes: int
    duplicate_node_ids: int
    invalid_edges: int
    missing_edge_endpoints: int


def inspect_graph(graph: dict) -> GraphInspection:
    if not isinstance(graph, dict):
        raise ValueError("Graph must be a JSON object")

    nodes = graph.get("nodes")
    edges = graph.get("edges")

    if not isinstance(nodes, list):
        raise ValueError("Graph nodes must be a list")

    if not isinstance(edges, list):
        raise ValueError("Graph edges must be a list")

    node_ids = set()
    node_labels = Counter()
    edge_types = Counter()

    invalid_nodes = 0
    duplicate_node_ids = 0
    invalid_edges = 0
    missing_edge_endpoints = 0

    for node in nodes:
        if not isinstance(node, dict):
            invalid_nodes += 1
            continue

        node_id = node.get("id")

        if type(node_id) is not int:
            invalid_nodes += 1
            continue

        if node_id in node_ids:
            duplicate_node_ids += 1

        node_ids.add(node_id)

        label = node.get("_label")

        if isinstance(label, str) and label.strip():
            node_labels[label] += 1
        else:
            node_labels["<missing>"] += 1

    for edge in edges:
        if not isinstance(edge, dict):
            invalid_edges += 1
            continue

        source = edge.get("outNode")
        target = edge.get("inNode")

        if type(source) is not int or type(target) is not int:
            invalid_edges += 1
            continue

        edge_type = edge.get("etype")

        if isinstance(edge_type, str) and edge_type.strip():
            edge_types[edge_type] += 1
        else:
            edge_types["<missing>"] += 1

        if source not in node_ids or target not in node_ids:
            missing_edge_endpoints += 1

    return GraphInspection(
        node_count=len(nodes),
        edge_count=len(edges),
        node_labels=dict(node_labels),
        edge_types=dict(edge_types),
        invalid_nodes=invalid_nodes,
        duplicate_node_ids=duplicate_node_ids,
        invalid_edges=invalid_edges,
        missing_edge_endpoints=missing_edge_endpoints,
    )


def inspect_graph_file(path: str | Path) -> GraphInspection:
    with Path(path).open("r", encoding="utf-8") as file:
        graph = json.load(file)

    return inspect_graph(graph)


if __name__ == "__main__":
    example_graph = {
        "nodes": [
            {"id": 1, "_label": "METHOD"},
            {"id": 2, "_label": "CALL"},
        ],
        "edges": [
            {"outNode": 1, "inNode": 2, "etype": "AST"},
        ],
    }

    result = inspect_graph(example_graph)

    assert result.node_count == 2
    assert result.edge_count == 1
    assert result.node_labels == {"METHOD": 1, "CALL": 1}
    assert result.edge_types == {"AST": 1}
    assert result.invalid_nodes == 0
    assert result.duplicate_node_ids == 0
    assert result.invalid_edges == 0
    assert result.missing_edge_endpoints == 0

    broken_graph = {
        "nodes": [{"id": 1, "_label": "METHOD"}],
        "edges": [
            {"outNode": 1, "inNode": 99, "etype": "AST"}
        ],
    }

    broken_result = inspect_graph(broken_graph)

    assert broken_result.missing_edge_endpoints == 1

    try:
        inspect_graph({"nodes": [], "edges": None})
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed graph was accepted")

    print("Valid graph inspection: PASSED")
    print("Missing endpoint detection: PASSED")
    print("Malformed graph rejection: PASSED")
    print("\nGRAPH CONTENT MODULE TEST PASSED")
