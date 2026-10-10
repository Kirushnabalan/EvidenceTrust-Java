import json
from pathlib import Path
from collections import defaultdict


JSON_PATH = Path("dataset/raw/megavul/megavul.json")


def main():
    with JSON_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    groups = defaultdict(
        lambda: {
            "total": 0,
            "vulnerable": 0,
            "non_vulnerable": 0,
            "commits": set(),
            "repos": set(),
        }
    )

    for record in records:
        cve = record["cve_id"]

        g = groups[cve]

        g["total"] += 1

        if record["is_vul"]:
            g["vulnerable"] += 1
        else:
            g["non_vulnerable"] += 1

        g["commits"].add(record["commit_hash"])
        g["repos"].add(record["repo_name"])

    print("=== CVE GROUP SUMMARY ===")
    print("Unique CVEs:", len(groups))
    print("Total records:", len(records))

    total_vul = sum(g["vulnerable"] for g in groups.values())
    total_non = sum(g["non_vulnerable"] for g in groups.values())

    print("Vulnerable:", total_vul)
    print("Non-vulnerable:", total_non)

    print("\n=== LARGEST CVE GROUPS ===")

    largest = sorted(groups.items(), key=lambda x: x[1]["total"], reverse=True)

    for cve, g in largest[:30]:
        print(
            f"{cve:20} "
            f"total={g['total']:5} "
            f"vul={g['vulnerable']:4} "
            f"non_vul={g['non_vulnerable']:5} "
            f"commits={len(g['commits']):3} "
            f"repos={len(g['repos']):2}"
        )

    print("\n=== VULNERABLE GROUP DISTRIBUTION ===")

    vulnerable_groups = [g for g in groups.values() if g["vulnerable"] > 0]

    print("CVE groups containing vulnerable records:", len(vulnerable_groups))

    print("Largest vulnerable group:", max(g["vulnerable"] for g in vulnerable_groups))

    print("Smallest vulnerable group:", min(g["vulnerable"] for g in vulnerable_groups))


if __name__ == "__main__":
    main()
