from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GraphNode:
    index: int
    joern_id: int
    label: str

    code: str | None
    name: str | None

    line_number: int | None
    column_number: int | None

    type_full_name: str | None
    method_full_name: str | None

    raw: dict[str, Any]


@dataclass
class GraphEdge:
    source: int
    target: int

    source_joern_id: int
    target_joern_id: int

    edge_type: str
    variable: Any

    raw: dict[str, Any]


@dataclass
class ParsedGraph:
    path: Path

    nodes: list[GraphNode]
    edges: list[GraphEdge]

    joern_id_to_index: dict[int, int]

    node_label_counts: Counter
    edge_type_counts: Counter

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


class GraphParseError(ValueError):
    pass


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)

    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None

    value = str(value)

    if not value.strip():
        return None

    return value


def parse_graph_dict(
    graph: dict[str, Any],
    path: Path,
) -> ParsedGraph:
    if not isinstance(graph, dict):
        raise GraphParseError(
            "Graph top level must be a dictionary."
        )

    nodes_raw = graph.get("nodes")
    edges_raw = graph.get("edges")

    if not isinstance(nodes_raw, list):
        raise GraphParseError(
            "'nodes' must be a list."
        )

    if not isinstance(edges_raw, list):
        raise GraphParseError(
            "'edges' must be a list."
        )

    nodes: list[GraphNode] = []
    joern_id_to_index: dict[int, int] = {}
    node_label_counts = Counter()

    for index, raw_node in enumerate(nodes_raw):
        if not isinstance(raw_node, dict):
            raise GraphParseError(
                f"Node {index} is not a dictionary."
            )

        joern_id = raw_node.get("id")

        if not isinstance(joern_id, int):
            raise GraphParseError(
                f"Node {index} has invalid id: "
                f"{joern_id!r}"
            )

        if joern_id in joern_id_to_index:
            raise GraphParseError(
                f"Duplicate Joern node id: {joern_id}"
            )

        label = raw_node.get("_label")

        if label is None:
            label = "UNKNOWN"

        label = str(label)

        node = GraphNode(
            index=index,
            joern_id=joern_id,
            label=label,

            code=_optional_str(
                raw_node.get("code")
            ),

            name=_optional_str(
                raw_node.get("name")
            ),

            line_number=_optional_int(
                raw_node.get("lineNumber")
            ),

            column_number=_optional_int(
                raw_node.get("columnNumber")
            ),

            type_full_name=_optional_str(
                raw_node.get("typeFullName")
            ),

            method_full_name=_optional_str(
                raw_node.get("methodFullName")
            ),

            raw=raw_node,
        )

        nodes.append(node)

        joern_id_to_index[
            joern_id
        ] = index

        node_label_counts[label] += 1

    edges: list[GraphEdge] = []
    edge_type_counts = Counter()

    for edge_index, raw_edge in enumerate(
        edges_raw
    ):
        if not isinstance(raw_edge, dict):
            raise GraphParseError(
                f"Edge {edge_index} is not "
                f"a dictionary."
            )

        source_joern_id = raw_edge.get(
            "outNode"
        )

        target_joern_id = raw_edge.get(
            "inNode"
        )

        if source_joern_id not in joern_id_to_index:
            raise GraphParseError(
                f"Edge {edge_index} has unknown "
                f"outNode={source_joern_id!r}"
            )

        if target_joern_id not in joern_id_to_index:
            raise GraphParseError(
                f"Edge {edge_index} has unknown "
                f"inNode={target_joern_id!r}"
            )

        edge_type = raw_edge.get("etype")

        if edge_type is None:
            edge_type = "UNKNOWN"

        edge_type = str(edge_type)

        edge = GraphEdge(
            source=joern_id_to_index[
                source_joern_id
            ],

            target=joern_id_to_index[
                target_joern_id
            ],

            source_joern_id=source_joern_id,
            target_joern_id=target_joern_id,

            edge_type=edge_type,

            variable=raw_edge.get(
                "variable"
            ),

            raw=raw_edge,
        )

        edges.append(edge)

        edge_type_counts[
            edge_type
        ] += 1

    return ParsedGraph(
        path=path,
        nodes=nodes,
        edges=edges,

        joern_id_to_index=joern_id_to_index,

        node_label_counts=node_label_counts,
        edge_type_counts=edge_type_counts,
    )


def load_graph(
    path: str | Path,
) -> ParsedGraph:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Graph file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        graph = json.load(f)

    return parse_graph_dict(
        graph=graph,
        path=path,
    )