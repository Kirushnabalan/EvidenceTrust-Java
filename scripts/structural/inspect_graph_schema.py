import csv
import json
from collections import Counter
from pathlib import Path

# This file is located at:
# scripts/structural/inspect_graph_schema.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (PROJECT_ROOT / "dataset" / "processed" / "megavul_manifest.csv")

MEGAVUL_ROOT = (PROJECT_ROOT / "dataset" / "raw" / "megavul")

# Path resolution
def resolve_graph_path(raw_path: str):
   # Resolve a graph path stored in the MegaVul manifest.
    if not raw_path:
        return None

    raw_path = raw_path.strip()
    path = Path(raw_path)

    candidates = [
        path,
        PROJECT_ROOT / path,
        MEGAVUL_ROOT / path,
        MEGAVUL_ROOT / "megavul_graph" / path,
        MEGAVUL_ROOT / "megavul_graph" / path.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None

# Structure preview
def preview_value(value, indent=0, max_items=3):
    prefix = " " * indent

    if isinstance(value, dict):
        print(f"{prefix}DICT ({len(value)} keys)")

        for key, child in list(value.items())[:20]:
            print(f"{prefix}  {key!r}: {type(child).__name__}")

            if isinstance(child, (dict, list)):
                preview_value(child,indent + 4,max_items,)

    elif isinstance(value, list):
        print(f"{prefix}LIST (length={len(value)})")

        for index, item in enumerate(value[:max_items]):
            print(f"{prefix}  [{index}] {type(item).__name__}")

            if isinstance(item, (dict, list)):
                preview_value(item,indent + 4,max_items,)

    else:
        text = repr(value)

        if len(text) > 250:
            text = text[:250] + "..."

        print(f"{prefix}{type(value).__name__}: {text}")

# Manifest loading
def load_manifest():
    with MANIFEST_PATH.open("r",encoding="utf-8",newline="",) as file:
        rows = list(csv.DictReader(file))

    # Use development data only.
    return [
        row
        for row in rows
        if row.get("split") in {"train", "validation"}
    ]

# Sample selection
def select_examples(rows):
    selected = []

    # Select 2 vulnerable and 2 non-vulnerable samples.
    for label in ("1", "0"):
        matches = [row for row in rows if row.get("is_vul") == label and row.get("graph_path")]

        selected.extend(matches[:2])

    return selected

# Graph inspection

def inspect_graph(row, number):
    print()
    print("-------" )
    print(f"SAMPLE {number}")
    print("-------" )

    print("sample_id:", row.get("sample_id"))
    print("split:", row.get("split"))
    print("is_vul:", row.get("is_vul"))
    print("manifest graph_path:", row.get("graph_path"))

    graph_path = resolve_graph_path(row.get("graph_path", ""))

    if graph_path is None:
        print("STATUS: GRAPH PATH COULD NOT BE RESOLVED")
        return None

    print("resolved path:", graph_path)
    print("suffix:", graph_path.suffix)
    print("size bytes:", graph_path.stat().st_size)

    try:
        with graph_path.open("r",encoding="utf-8",) as file:
            graph = json.load(file)

    except UnicodeDecodeError as exc:
        print("STATUS: NOT UTF-8 TEXT")
        print("error:", repr(exc))
        return None

    except json.JSONDecodeError as exc:
        print("STATUS: NOT STANDARD JSON")
        print("error:", repr(exc))

        # Show the beginning of the file for format inspection.
        try:
            with graph_path.open("r",encoding="utf-8",errors="replace",) as file:
                preview = file.read(1000)

            print()
            print("RAW FILE PREVIEW")
            print("-------" )
            print(preview)

        except Exception as preview_exc:
            print("Preview failed:", repr(preview_exc))

        return None

    except Exception as exc:
        print("STATUS: LOAD FAILED")
        print("error:", repr(exc))
        return None

    print()
    print("top-level type:", type(graph).__name__)

    if isinstance(graph, dict):
        print("top-level keys:", list(graph.keys()))

    elif isinstance(graph, list):
        print("top-level list length:", len(graph))

        if graph:
            print("first item type:", type(graph[0]).__name__)

            if isinstance(graph[0], dict):
                print("first item keys:", list(graph[0].keys()))

    print()
    print("STRUCTURE PREVIEW")
    print("------------")

    preview_value(graph)

    return graph

# Main

def main():
    print("Project root:", PROJECT_ROOT)
    print("Manifest:", MANIFEST_PATH)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    rows = load_manifest()

    print("Development samples:", f"{len(rows):,}")

    selected = select_examples(rows)

    if not selected:
        raise RuntimeError("No graph samples found.")

    signatures = Counter()

    for index, row in enumerate(selected, start=1):
        graph = inspect_graph(row,index,)

        if isinstance(graph, dict):
            signature = tuple(sorted(graph.keys()))

            signatures[("dict",) + signature] += 1

        elif (isinstance(graph, list) and graph and isinstance(graph[0], dict)):
            signature = tuple(sorted(graph[0].keys()))

            signatures[("list",) + signature] += 1

    print()
    print("--------")
    print("SCHEMA SIGNATURE SUMMARY")
    print("--------")

    for signature, count in signatures.most_common():
        print()
        print("count:", count)
        print("signature:", signature)


if __name__ == "__main__":
    main()