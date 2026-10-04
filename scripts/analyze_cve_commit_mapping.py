import json
from pathlib import Path
from collections import defaultdict


JSON_PATH = Path("dataset/raw/megavul/megavul.json")


def main():
    with JSON_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    cve_to_commits = defaultdict(set)
    commit_to_cves = defaultdict(set)

    for record in records:
        repo = record.get("repo_name")
        commit = record.get("commit_hash")
        cve = record.get("cve_id")

        if not repo or not commit or not cve:
            continue

        commit_key = (repo, commit)
        cve_key = (repo, cve)

        cve_to_commits[cve_key].add(commit_key)
        commit_to_cves[commit_key].add(cve_key)

    multi_commit_cves = {
        cve: commits for cve, commits in cve_to_commits.items() if len(commits) > 1
    }

    multi_cve_commits = {
        commit: cves for commit, cves in commit_to_cves.items() if len(cves) > 1
    }

    print("=== CVE → REPO+COMMIT MAPPING ===")
    print("CVE groups:", len(cve_to_commits))

    print("CVEs mapped to multiple repo+commit groups:", len(multi_commit_cves))

    print()

    for cve, commits in list(multi_commit_cves.items())[:20]:
        print("CVE:", cve)

        for repo, commit in sorted(commits):
            print("   ", repo, commit)

    print()
    print("=== REPO+COMMIT → CVE MAPPING ===")

    print("Repo+commit groups:", len(commit_to_cves))

    print("Repo+commit groups containing multiple CVEs:", len(multi_cve_commits))

    print()

    for (repo, commit), cves in list(multi_cve_commits.items())[:20]:
        print("Repository:", repo)
        print("Commit:", commit)

        for _, cve in sorted(cves):
            print("   ", cve)


if __name__ == "__main__":
    main()
