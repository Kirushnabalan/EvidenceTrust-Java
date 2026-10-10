import json
from pathlib import Path
from collections import defaultdict


JSON_PATH = Path("dataset/raw/megavul/megavul.json")


def main():
    with JSON_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    before_map = defaultdict(list)
    after_map = defaultdict(list)

    for idx, record in enumerate(records):
        before = record.get("func_graph_path_before")
        after = record.get("func_graph_path")

        if before:
            before_map[before].append((idx, record))

        if after:
            after_map[after].append((idx, record))

    print("=== REPEATED BEFORE GRAPH PATHS ===")

    count = 0

    for path, items in before_map.items():
        if len(items) > 1:
            count += 1

            print("\nGRAPH:", path)

            for idx, record in items:
                print(
                    "  index=",
                    idx,
                    "cve=",
                    record.get("cve_id"),
                    "repo=",
                    record.get("repo_name"),
                    "commit=",
                    record.get("commit_hash"),
                    "func=",
                    record.get("func_name"),
                    "is_vul=",
                    record.get("is_vul"),
                )

            if count >= 20:
                break

    print("\nRepeated before paths shown:", count)

    print("\n=== REPEATED AFTER GRAPH PATHS ===")

    count = 0

    for path, items in after_map.items():
        if len(items) > 1:
            count += 1

            print("\nGRAPH:", path)

            for idx, record in items[:10]:
                print(
                    "  index=",
                    idx,
                    "cve=",
                    record.get("cve_id"),
                    "repo=",
                    record.get("repo_name"),
                    "commit=",
                    record.get("commit_hash"),
                    "func=",
                    record.get("func_name"),
                    "is_vul=",
                    record.get("is_vul"),
                )

            if count >= 20:
                break

    print("\nRepeated after paths shown:", count)


if __name__ == "__main__":
    main()
