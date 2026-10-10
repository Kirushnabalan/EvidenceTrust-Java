import json
from pathlib import Path
from collections import Counter


JSON_PATH = Path("dataset/raw/megavul/megavul.json")


def main():
    with JSON_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    repos = Counter()
    commits = Counter()
    cves = Counter()

    for record in records:
        repos[record.get("repo_name")] += 1

        commit_key = (record.get("repo_name"), record.get("commit_hash"))
        commits[commit_key] += 1

        cve = record.get("cve_id")
        if cve:
            cves[cve] += 1

    print("-- DATASET --")
    print(f"Records: {len(records):,}")

    print("\n-- REPOSITORIES --")
    print(f"Unique repositories: {len(repos):,}")

    print("\n-- COMMITS --")
    print(f"Unique repo+commit groups: {len(commits):,}")

    print("\n-- CVEs --")
    print(f"Unique CVEs: {len(cves):,}")

    print("\n-- LARGEST REPOSITORIES --")

    for repo, count in repos.most_common(20):
        print(f"{count:6}  {repo}")

    print("\n-- LARGEST REPO+COMMIT GROUPS --")

    for (repo, commit), count in commits.most_common(20):
        print(f"{count:6}  {repo}  {commit}")

    print("\n-- LARGEST CVE GROUPS --")

    for cve, count in cves.most_common(20):
        print(f"{count:6}  {cve}")


if __name__ == "__main__":
    main()
