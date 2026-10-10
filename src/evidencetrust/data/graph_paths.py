
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


@dataclass(frozen=True)
class GraphPathResult:
    reference: str
    resolved_path: Path
    exists: bool


def check_graph_reference(
    graph_root: str | Path,
    graph_reference: str,
) -> GraphPathResult:

    if not isinstance(graph_reference, str) or not graph_reference.strip():
        raise ValueError("Graph reference is empty or invalid")

    normalized = graph_reference.strip().replace("\\", "/")

    if normalized.startswith("/") or ":" in normalized:
        raise ValueError("Graph reference must be a relative path")

    parts = normalized.split("/")

    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("Graph reference contains unsafe path components")

    root = Path(graph_root).resolve()
    graph_path = root.joinpath(*parts).resolve()

    if not graph_path.is_relative_to(root):
        raise ValueError("Graph reference escapes the graph directory")

    return GraphPathResult(
        reference=graph_reference,
        resolved_path=graph_path,
        exists=graph_path.is_file(),
    )


if __name__ == "__main__":

    with TemporaryDirectory() as directory:

        graph_root = Path(directory)

        test_file = graph_root / "example" / "0.json"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("{}", encoding="utf-8")

        found = check_graph_reference(
            graph_root,
            "example/0.json",
        )

        missing = check_graph_reference(
            graph_root,
            "example/missing.json",
        )

        assert found.exists
        assert not missing.exists

        print("Existing graph:", "FOUND")
        print("Missing graph:", "NOT FOUND")

        try:
            check_graph_reference(graph_root, "../outside.json")
        except ValueError:
            print("Unsafe graph reference: REJECTED")
        else:
            raise AssertionError("Unsafe path was accepted")

    print("\nGRAPH PATH MODULE TEST PASSED")
