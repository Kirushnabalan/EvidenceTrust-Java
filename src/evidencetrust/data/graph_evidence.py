
from dataclasses import dataclass
from pathlib import Path

from .graph_content import GraphInspection, inspect_graph_file


@dataclass(frozen=True)
class GraphEvidenceSummary:
    node_count: int
    edge_count: int
    ast_edges: int
    cfg_edges: int
    reaching_def_edges: int
    cdg_edges: int
    call_nodes: int
    invalid_nodes: int
    duplicate_node_ids: int
    invalid_edges: int
    missing_edge_endpoints: int


def summarize_graph(
    inspection: GraphInspection,
) -> GraphEvidenceSummary:

    edges = inspection.edge_types
    nodes = inspection.node_labels

    return GraphEvidenceSummary(
        node_count=inspection.node_count,
        edge_count=inspection.edge_count,
        ast_edges=edges.get("AST", 0),
        cfg_edges=edges.get("CFG", 0),
        reaching_def_edges=edges.get("REACHING_DEF", 0),
        cdg_edges=edges.get("CDG", 0),
        call_nodes=nodes.get("CALL", 0),
        invalid_nodes=inspection.invalid_nodes,
        duplicate_node_ids=inspection.duplicate_node_ids,
        invalid_edges=inspection.invalid_edges,
        missing_edge_endpoints=inspection.missing_edge_endpoints,
    )


def summarize_graph_file(
    path: str | Path,
) -> GraphEvidenceSummary:

    inspection = inspect_graph_file(path)

    return summarize_graph(inspection)


if __name__ == "__main__":
    import json
    from tempfile import TemporaryDirectory

    example = {
        "nodes": [
            {"id": 1, "_label": "METHOD"},
            {"id": 2, "_label": "CALL"},
            {"id": 3, "_label": "IDENTIFIER"},
        ],
        "edges": [
            {"outNode": 1, "inNode": 2, "etype": "AST"},
            {"outNode": 1, "inNode": 2, "etype": "CFG"},
            {"outNode": 3, "inNode": 2, "etype": "REACHING_DEF"},
        ],
    }

    with TemporaryDirectory() as directory:
        path = Path(directory) / "example.json"

        path.write_text(
            json.dumps(example),
            encoding="utf-8",
        )

        result = summarize_graph_file(path)

        assert result.node_count == 3
        assert result.edge_count == 3
        assert result.ast_edges == 1
        assert result.cfg_edges == 1
        assert result.reaching_def_edges == 1
        assert result.cdg_edges == 0
        assert result.call_nodes == 1

        assert result.invalid_nodes == 0
        assert result.duplicate_node_ids == 0
        assert result.invalid_edges == 0
        assert result.missing_edge_endpoints == 0

    print("Graph size extraction: PASSED")
    print("Structural relationship extraction: PASSED")
    print("Integrity statistics extraction: PASSED")
    print("\nGRAPH EVIDENCE MODULE TEST PASSED")
