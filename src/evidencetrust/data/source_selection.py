
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SelectedEvidence:
    label: int
    source_code: str
    graph_reference: str
    source_field: str
    graph_field: str


def select_evidence(record: Mapping[str, Any]) -> SelectedEvidence:
    label = record.get("is_vul")

    if not isinstance(label, bool):
        raise ValueError("Expected is_vul to be a boolean")

    if label:
        source_field = "func_before"
        graph_field = "func_graph_path_before"
    else:
        source_field = "func"
        graph_field = "func_graph_path"

    source_code = record.get(source_field)
    graph_reference = record.get(graph_field)

    if not isinstance(source_code, str) or not source_code.strip():
        raise ValueError(f"Missing source code: {source_field}")

    if not isinstance(graph_reference, str) or not graph_reference.strip():
        raise ValueError(f"Missing graph reference: {graph_field}")

    return SelectedEvidence(
        label=int(label),
        source_code=source_code,
        graph_reference=graph_reference,
        source_field=source_field,
        graph_field=graph_field,
    )


if __name__ == "__main__":
    examples = [
        {
            "is_vul": True,
            "func_before": "void vulnerable() {}",
            "func_graph_path_before": "before/0.json",
        },
        {
            "is_vul": False,
            "func": "void safe() {}",
            "func_graph_path": "normal/0.json",
        },
    ]

    for record in examples:
        evidence = select_evidence(record)

        print(
            f"Label: {evidence.label} | "
            f"Source: {evidence.source_field} | "
            f"Graph: {evidence.graph_field}"
        )

    print("SOURCE SELECTION TEST PASSED")
