import json
from collections import Counter
from pathlib import Path

JSON_PATH = Path("dataset/raw/megavul/megavul.json")
GRAPH_ROOT = Path("dataset/raw/megavul/megavul_graph")


def main():
    print("Loading MegaVul JSON...")

    with JSON_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Records: {len(records):,}")

    before_paths = []
    after_paths = []

    missing_before = []
    missing_after = []

    for i, record in enumerate(records):
        before = record.get("func_graph_path_before")
        after = record.get("func_graph_path")

        if before:
            before_paths.append(before)
            path = GRAPH_ROOT / before
            if not path.exists():
                missing_before.append((i, before))

        if after:
            after_paths.append(after)
            path = GRAPH_ROOT / after
            if not path.exists():
                missing_after.append((i, after))

    print("\n-- GRAPH PATH COUNTS --")
    print(f"Records with before graph path: {len(before_paths):,}")
    print(f"Unique before graph paths:     {len(set(before_paths)):,}")
    print(f"Records with after graph path:  {len(after_paths):,}")
    print(f"Unique after graph paths:       {len(set(after_paths)):,}")

    print("\n-- MISSING GRAPH FILES --")
    print(f"Missing before graphs: {len(missing_before):,}")
    print(f"Missing after graphs:  {len(missing_after):,}")

    print("\n-- DUPLICATE PATHS --")
    before_counter = Counter(before_paths)
    after_counter = Counter(after_paths)

    duplicate_before = sum(1 for count in before_counter.values() if count > 1)
    duplicate_after = sum(1 for count in after_counter.values() if count > 1)

    print(f"Repeated before paths: {duplicate_before:,}")
    print(f"Repeated after paths:  {duplicate_after:,}")

    print("\n-- LABEL COUNTS --")
    labels = Counter(record.get("is_vul") for record in records)
    for label, count in labels.items():
        print(f"is_vul={label}: {count:,}")

    print("\n-- CWE INFORMATION --")
    records_with_cwe = sum(1 for record in records if record.get("cwe_ids"))
    print(f"Records with CWE: {records_with_cwe:,}")
    print(f"Records without CWE: {len(records) - records_with_cwe:,}")

    print("\n-- LOCALIZATION INFORMATION --")
    records_with_diff = sum(1 for record in records if record.get("diff_line_info"))
    print(f"Records with diff_line_info: {records_with_diff:,}")
    print(f"Records without diff_line_info: {len(records) - records_with_diff:,}")


if __name__ == "__main__":
    main()
