import json
import hashlib
from pathlib import Path
from collections import Counter


JSON_PATH = Path("dataset/raw/megavul/megavul.json")


def code_hash(code):
    if code is None:
        return None

    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def main():

    with JSON_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Total records: {len(records):,}")

    identities = []

    for record in records:
        if record.get("is_vul") is True:
            graph_path = record.get("func_graph_path_before")
            code = record.get("func_before")
        else:
            graph_path = record.get("func_graph_path")
            code = record.get("func")

        identity = (
            record.get("repo_name"),
            record.get("commit_hash"),
            record.get("file_path"),
            record.get("func_name"),
            graph_path,
            code_hash(code),
            record.get("is_vul"),
        )

        identities.append(identity)

    counts = Counter(identities)

    duplicate_groups = {key: count for key, count in counts.items() if count > 1}

    duplicate_rows = sum(count - 1 for count in duplicate_groups.values())

    print()
    print("=== EXACT DUPLICATE ANALYSIS ===")
    print(f"Unique identities:       {len(counts):,}")
    print(f"Duplicate groups:        {len(duplicate_groups):,}")
    print(f"Duplicate extra records: {duplicate_rows:,}")

    print()
    print("=== EXAMPLES ===")

    shown = 0

    for identity, count in duplicate_groups.items():
        print()
        print("Occurrences:", count)
        print("Repository:", identity[0])
        print("Commit:", identity[1])
        print("File:", identity[2])
        print("Function:", identity[3])
        print("Graph:", identity[4])
        print("is_vul:", identity[6])

        shown += 1

        if shown >= 20:
            break


if __name__ == "__main__":
    main()
