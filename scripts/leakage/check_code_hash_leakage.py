import csv
from collections import defaultdict
from pathlib import Path

MANIFEST = Path("dataset/processed/megavul_manifest.csv")

hash_to_records = defaultdict(list)

with MANIFEST.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        code_hash = row["code_hash"]

        hash_to_records[code_hash].append(
            {
                "sample_id": row["sample_id"],
                "cve": row["cve_id"],
                "repo": row["repo_name"],
                "label": row["is_vul"],
                "split": row["split"],
                "function": row["func_name"],
            }
        )


multi_cve = []
multi_label = []

for code_hash, records in hash_to_records.items():
    cves = set(r["cve"] for r in records)
    labels = set(r["label"] for r in records)

    if len(cves) > 1:
        multi_cve.append((code_hash, records))

    if len(labels) > 1:
        multi_label.append((code_hash, records))


print("-- CODE HASH AUDIT --")

print("Unique code hashes:", len(hash_to_records))

print("Code hashes appearing under multiple CVEs:", len(multi_cve))

print("Code hashes appearing with multiple labels:", len(multi_label))


print("\n-- EXAMPLES: MULTIPLE CVEs --")

for code_hash, records in multi_cve[:20]:
    print("\nHASH:", code_hash)

    for r in records:
        print(
            " ",
            r["sample_id"],
            r["cve"],
            r["repo"],
            "label=" + r["label"],
            r["function"],
        )


print("\n-- EXAMPLES: MULTIPLE LABELS --")

for code_hash, records in multi_label[:20]:
    print("\nHASH:", code_hash)

    for r in records:
        print(
            " ",
            r["sample_id"],
            r["cve"],
            r["repo"],
            "label=" + r["label"],
            r["function"],
        )
