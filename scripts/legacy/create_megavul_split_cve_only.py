import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


RAW_JSON = Path("dataset/raw/megavul/megavul.json")
OUTPUT_DIR = Path("dataset/splits")
MANIFEST_PATH = Path("dataset/processed/megavul_manifest.csv")

SEED = 42

TARGETS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}


def code_hash(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def load_records():
    with RAW_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_groups(records):
    groups = defaultdict(list)

    for index, record in enumerate(records):
        cve = record.get("cve_id")

        if not cve:
            raise ValueError(f"Record {index} has no CVE ID.")

        groups[cve].append((index, record))

    return groups


def group_statistics(groups):

    stats = {}

    for cve, items in groups.items():
        vulnerable = sum(1 for _, r in items if r["is_vul"])

        non_vulnerable = len(items) - vulnerable

        stats[cve] = {
            "total": len(items),
            "vulnerable": vulnerable,
            "non_vulnerable": non_vulnerable,
        }

    return stats


def choose_split(
    totals,
    vulnerable,
    non_vulnerable,
    group_counts,
    target_totals,
    target_vulnerable,
    target_non_vulnerable,
    target_groups,
):
    best_split = None
    best_score = None

    for split in TARGETS:
        new_total = totals[split]
        new_vulnerable = vulnerable[split]
        new_non_vulnerable = non_vulnerable[split]
        new_groups = group_counts[split]

        # Normalized deviations after assigning this group.
        total_error = ((new_total - target_totals[split]) / target_totals["train"]) ** 2

        vulnerable_error = (
            (new_vulnerable - target_vulnerable[split]) / target_vulnerable["train"]
        ) ** 2

        non_vulnerable_error = (
            (new_non_vulnerable - target_non_vulnerable[split])
            / target_non_vulnerable["train"]
        ) ** 2

        group_error = (
            (new_groups - target_groups[split]) / target_groups["train"]
        ) ** 2

        score = (
            0.45 * total_error
            + 0.40 * vulnerable_error
            + 0.10 * non_vulnerable_error
            + 0.05 * group_error
        )

        if best_score is None or score < best_score:
            best_score = score
            best_split = split

    return best_split


def create_assignment(groups, stats):

    rng = random.Random(SEED)

    cves = list(groups.keys())

    # Largest groups first.
    cves.sort(key=lambda cve: stats[cve]["total"], reverse=True)

    # Randomize only equal-size groups so the result
    # does not depend on dictionary ordering.
    i = 0

    while i < len(cves):
        j = i + 1

        while j < len(cves) and stats[cves[j]]["total"] == stats[cves[i]]["total"]:
            j += 1

        block = cves[i:j]
        rng.shuffle(block)
        cves[i:j] = block

        i = j

    total_records = sum(x["total"] for x in stats.values())

    total_vulnerable = sum(x["vulnerable"] for x in stats.values())

    total_non_vulnerable = sum(x["non_vulnerable"] for x in stats.values())

    total_groups = len(cves)

    target_totals = {
        split: total_records * fraction for split, fraction in TARGETS.items()
    }

    target_vulnerable = {
        split: total_vulnerable * fraction for split, fraction in TARGETS.items()
    }

    target_non_vulnerable = {
        split: total_non_vulnerable * fraction for split, fraction in TARGETS.items()
    }

    target_groups = {
        split: total_groups * fraction for split, fraction in TARGETS.items()
    }

    assignments = {}

    totals = {split: 0 for split in TARGETS}

    vulnerable = {split: 0 for split in TARGETS}

    non_vulnerable = {split: 0 for split in TARGETS}

    group_counts = {split: 0 for split in TARGETS}

    for cve in cves:
        s = stats[cve]

        # Temporarily add the group to each candidate.
        candidate_totals = totals.copy()
        candidate_vulnerable = vulnerable.copy()
        candidate_non_vulnerable = non_vulnerable.copy()
        candidate_group_counts = group_counts.copy()

        best_split = None
        best_score = None

        for split in TARGETS:
            candidate_totals[split] += s["total"]
            candidate_vulnerable[split] += s["vulnerable"]
            candidate_non_vulnerable[split] += s["non_vulnerable"]
            candidate_group_counts[split] += 1

            score = (
                0.45
                * (
                    (candidate_totals[split] - target_totals[split])
                    / target_totals["train"]
                )
                ** 2
                + 0.40
                * (
                    (candidate_vulnerable[split] - target_vulnerable[split])
                    / target_vulnerable["train"]
                )
                ** 2
                + 0.10
                * (
                    (candidate_non_vulnerable[split] - target_non_vulnerable[split])
                    / target_non_vulnerable["train"]
                )
                ** 2
                + 0.05
                * (
                    (candidate_group_counts[split] - target_groups[split])
                    / target_groups["train"]
                )
                ** 2
            )

            candidate_totals[split] -= s["total"]
            candidate_vulnerable[split] -= s["vulnerable"]
            candidate_non_vulnerable[split] -= s["non_vulnerable"]
            candidate_group_counts[split] -= 1

            if best_score is None or score < best_score:
                best_score = score
                best_split = split

        assignments[cve] = best_split

        totals[best_split] += s["total"]
        vulnerable[best_split] += s["vulnerable"]
        non_vulnerable[best_split] += s["non_vulnerable"]
        group_counts[best_split] += 1

    return assignments


def create_manifest(records, assignments):

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "sample_id",
        "split",
        "group_id",
        "cve_id",
        "repo_name",
        "commit_hash",
        "parent_commit_hash",
        "file_path",
        "func_name",
        "is_vul",
        "cwe_ids",
        "code_version",
        "graph_path",
        "diff_line_info",
        "code_hash",
    ]

    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        writer.writeheader()

        for index, record in enumerate(records):
            is_vul = bool(record["is_vul"])

            if is_vul:
                code = record["func_before"]
                graph_path = record["func_graph_path_before"]
                code_version = "before"
            else:
                code = record["func"]
                graph_path = record["func_graph_path"]
                code_version = "clean"

            writer.writerow(
                {
                    "sample_id": index,
                    "split": assignments[record["cve_id"]],
                    "group_id": record["cve_id"],
                    "cve_id": record["cve_id"],
                    "repo_name": record["repo_name"],
                    "commit_hash": record["commit_hash"],
                    "parent_commit_hash": record["parent_commit_hash"],
                    "file_path": record["file_path"],
                    "func_name": record["func_name"],
                    "is_vul": int(is_vul),
                    "cwe_ids": "|".join(record.get("cwe_ids", [])),
                    "code_version": code_version,
                    "graph_path": graph_path,
                    "diff_line_info": json.dumps(record.get("diff_line_info")),
                    "code_hash": code_hash(code),
                }
            )


def verify_split(records, assignments):

    split_cves = defaultdict(set)

    for record in records:
        cve = record["cve_id"]
        split = assignments[cve]
        split_cves[split].add(cve)

    train = split_cves["train"]
    validation = split_cves["validation"]
    test = split_cves["test"]

    assert not train & validation
    assert not train & test
    assert not validation & test

    print("\n=== LEAKAGE CHECK ===")
    print("Train ∩ Validation:", len(train & validation))
    print("Train ∩ Test:", len(train & test))
    print("Validation ∩ Test:", len(validation & test))

    assert len(train & validation) == 0
    assert len(train & test) == 0
    assert len(validation & test) == 0

    print("CVE leakage check: PASSED")


def print_summary(records, assignments):

    summary = {
        split: {
            "records": 0,
            "vulnerable": 0,
            "non_vulnerable": 0,
            "cves": set(),
        }
        for split in TARGETS
    }

    for record in records:
        split = assignments[record["cve_id"]]

        summary[split]["records"] += 1
        summary[split]["cves"].add(record["cve_id"])

        if record["is_vul"]:
            summary[split]["vulnerable"] += 1
        else:
            summary[split]["non_vulnerable"] += 1

    print("\n=== FINAL SPLIT SUMMARY ===")

    for split in TARGETS:
        s = summary[split]

        print(f"\n{split.upper()}")

        print("  CVE groups:", len(s["cves"]))

        print("  Records:", s["records"])

        print("  Vulnerable:", s["vulnerable"])

        print("  Non-vulnerable:", s["non_vulnerable"])


def main():

    print("Loading MegaVul...")

    records = load_records()

    print(f"Loaded {len(records):,} records.")

    groups = build_groups(records)

    stats = group_statistics(groups)

    print(f"Found {len(groups):,} CVE groups.")

    assignments = create_assignment(groups, stats)

    print_summary(records, assignments)

    verify_split(records, assignments)

    create_manifest(records, assignments)

    print("\nManifest created:")
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
