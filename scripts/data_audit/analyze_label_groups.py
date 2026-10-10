import json
from pathlib import Path
from collections import defaultdict


JSON_PATH = Path("dataset/raw/megavul/megavul.json")


def main():
    with JSON_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    repo_stats = defaultdict(lambda: [0, 0])
    cve_stats = defaultdict(lambda: [0, 0])
    commit_stats = defaultdict(lambda: [0, 0])

    for record in records:
        label = int(record["is_vul"])

        repo = record.get("repo_name")
        cve = record.get("cve_id")
        commit = (record.get("repo_name"), record.get("commit_hash"))

        repo_stats[repo][label] += 1

        if cve:
            cve_stats[cve][label] += 1

        commit_stats[commit][label] += 1

    print("-- REPOSITORY LABEL DISTRIBUTION --")

    vulnerable_repos = 0

    for repo, counts in sorted(repo_stats.items(), key=lambda x: x[1][1], reverse=True):
        non_vul, vul = counts

        if vul > 0:
            vulnerable_repos += 1
            print(f"{repo:50} vulnerable={vul:5} non_vulnerable={non_vul:5}")

    print()
    print("Repositories with vulnerable samples:", vulnerable_repos)

    print("\n-- TOP CVEs BY VULNERABLE RECORDS --")

    for cve, counts in sorted(cve_stats.items(), key=lambda x: x[1][1], reverse=True)[
        :30
    ]:
        non_vul, vul = counts

        print(f"{cve:20} vulnerable={vul:5} non_vulnerable={non_vul:5}")

    print("\n-- REPO+COMMIT GROUPS CONTAINING VULNERABLE RECORDS --")

    vulnerable_commits = 0

    for (repo, commit), counts in sorted(
        commit_stats.items(), key=lambda x: x[1][1], reverse=True
    ):
        non_vul, vul = counts

        if vul > 0:
            vulnerable_commits += 1

            if vulnerable_commits <= 30:
                print(
                    f"{repo:40} {commit} vulnerable={vul:4} non_vulnerable={non_vul:4}"
                )

    print()
    print("Repo+commit groups containing vulnerable samples:", vulnerable_commits)


if __name__ == "__main__":
    main()
