import json
import hashlib
from pathlib import Path
from collections import defaultdict


JSON_PATH = Path("dataset/raw/megavul/megavul.json")


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def code_hash(code):

    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def main():

    print("Loading MegaVul...")

    with JSON_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    print("Records:", len(records))

    uf = UnionFind(len(records))

    cve_owner = {}
    code_owner = {}

    record_hashes = []

    # Build relationships

    for i, record in enumerate(records):
        cve = record.get("cve_id")

        if record["is_vul"]:
            code = record.get("func_before", "")
        else:
            code = record.get("func", "")

        h = code_hash(code)

        record_hashes.append(h)

        # Same CVE -> same leakage group
        if cve in cve_owner:
            uf.union(i, cve_owner[cve])
        else:
            cve_owner[cve] = i

        # Same exact code -> same leakage group
        if h in code_owner:
            uf.union(i, code_owner[h])
        else:
            code_owner[h] = i

    # Build components

    components = defaultdict(list)

    for i in range(len(records)):
        root = uf.find(i)
        components[root].append(i)

    print("\n-- LEAKAGE COMPONENT SUMMARY --")

    print("Records:", len(records))

    print("Unique CVEs:", len(cve_owner))

    print("Unique code hashes:", len(code_owner))

    print("Leakage components:", len(components))

    # Component statistics

    component_stats = []

    multi_cve_components = 0
    mixed_label_components = 0

    for root, indices in components.items():
        cves = set()
        hashes = set()
        repos = set()

        vulnerable = 0
        non_vulnerable = 0

        for i in indices:
            record = records[i]

            cves.add(record["cve_id"])
            hashes.add(record_hashes[i])
            repos.add(record["repo_name"])

            if record["is_vul"]:
                vulnerable += 1
            else:
                non_vulnerable += 1

        if len(cves) > 1:
            multi_cve_components += 1

        if vulnerable > 0 and non_vulnerable > 0:
            mixed_label_components += 1

        component_stats.append(
            {
                "root": root,
                "records": len(indices),
                "vulnerable": vulnerable,
                "non_vulnerable": non_vulnerable,
                "cves": cves,
                "hashes": hashes,
                "repos": repos,
            }
        )

    print("Components containing multiple CVEs:", multi_cve_components)

    print("Components containing both labels:", mixed_label_components)

    # Largest components

    print("\n-- LARGEST LEAKAGE COMPONENTS --")

    component_stats.sort(key=lambda x: x["records"], reverse=True)

    for number, component in enumerate(component_stats[:30], start=1):
        print(
            f"{number:2}. "
            f"records={component['records']:5} "
            f"vul={component['vulnerable']:4} "
            f"non_vul={component['non_vulnerable']:5} "
            f"CVEs={len(component['cves']):3} "
            f"hashes={len(component['hashes']):3} "
            f"repos={len(component['repos']):2}"
        )

        if len(component["cves"]) > 1:
            print("    CVEs:", ", ".join(sorted(component["cves"])[:10]))

    # Distribution

    sizes = [x["records"] for x in component_stats]

    print("\n-- COMPONENT SIZE DISTRIBUTION --")

    print("Largest:", max(sizes))

    print("Smallest:", min(sizes))

    print("Components:", len(sizes))

    print("Components with >1 record:", sum(1 for x in sizes if x > 1))


if __name__ == "__main__":
    main()
