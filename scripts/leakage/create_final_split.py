from __future__ import annotations

import csv
import hashlib
import json
import random
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

# PROJECT ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.data.java_comments import (
    remove_java_comments_preserve_layout,
)

from src.semantic.dataset import (
    MAX_CHUNKS,
    MODEL_MAX_LENGTH,
    OVERLAP,
)

# PATHS
RAW_JSON = PROJECT_ROOT / "dataset" / "raw" / "megavul" / "megavul.json"

MANIFEST = PROJECT_ROOT / "dataset" / "processed" / "megavul_manifest.csv"

SPLIT_DIR = PROJECT_ROOT / "dataset" / "splits"

LEAKAGE_REPORT = ( PROJECT_ROOT / "dataset" / "processed" / "final_split_leakage_report.json")

# CONFIGURATION
MODEL_NAME = "microsoft/codebert-base"

SEED = 42

SPLITS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

# Must match MegaVulSemanticDataset.
SPECIAL_TOKENS = 2

CONTENT_SIZE = MODEL_MAX_LENGTH - SPECIAL_TOKENS

STRIDE = CONTENT_SIZE - OVERLAP

# Therefore tokens beyond this capacity are invisible
# to the semantic model.
SEMANTIC_VISIBLE_CAPACITY = CONTENT_SIZE + (MAX_CHUNKS - 1) * STRIDE


# UNION FIND
class UnionFind:
    # Disjoint-set / union-find structure used to construct transitive leakage components.

    def __init__(self,n: int,) -> None:

        self.parent = list(range(n))

        self.size = [1] * n

    def find(self,x: int,) -> int:

        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]

        return x

    def union(self,a: int,b: int,) -> None:

        root_a = self.find(a)

        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = (
                root_b,
                root_a,
            )

        self.parent[root_b] = root_a

        self.size[root_a] += self.size[root_b]


# RAW CODE SELECTION
def get_code(record: dict[str, Any],) -> str:

    if bool(record.get("is_vul")):
        code = record.get("func_before","",)

    else:
        code = record.get("func","",)

    if not isinstance(code,str,):
        raise TypeError("Selected code is not a string.")

    if not code.strip():
        raise ValueError("Selected code is empty.")

    return code

# RAW CODE HASH
def code_hash(code: str,) -> str:
    # Exact raw selected-code SHA-256.
    return hashlib.sha256(code.encode("utf-8")).hexdigest()

# TOKENIZATION
def tokenize_semantic_code(tokenizer,code: str,) -> list[int]:

    cleaned_code = remove_java_comments_preserve_layout(code)

    if not cleaned_code.strip():
        raise ValueError("Code became empty after comment removal.")

    tokens = tokenizer.tokenize(cleaned_code)

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    if not isinstance(token_ids,list,):
        token_ids = list(token_ids)

    token_ids = [int(token_id) for token_id in token_ids]

    if not token_ids:
        raise ValueError("Semantic tokenization produced zero tokens.")

    return token_ids


# SEMANTIC MODEL-INPUT HASH
def semantic_input_hash(code: str,tokenizer,) -> str:

    token_ids = tokenize_semantic_code(tokenizer,code,)

    visible_ids = token_ids[:SEMANTIC_VISIBLE_CAPACITY]

    digest = hashlib.sha256()
    # Include configuration identity

    config = (
        f"model={MODEL_NAME};"
        f"max_length={MODEL_MAX_LENGTH};"
        f"special_tokens={SPECIAL_TOKENS};"
        f"content_size={CONTENT_SIZE};"
        f"overlap={OVERLAP};"
        f"stride={STRIDE};"
        f"max_chunks={MAX_CHUNKS};"
        f"bos={tokenizer.bos_token_id};"
        f"eos={tokenizer.eos_token_id};"
        f"pad={tokenizer.pad_token_id};"
    )

    digest.update(config.encode("utf-8"))

    digest.update(struct.pack("<I",len(visible_ids),))

    # Hash token IDs efficiently.
    if visible_ids:
        packed_ids = struct.pack(f"<{len(visible_ids)}I",*visible_ids,)

        digest.update(packed_ids)

    return digest.hexdigest()


# LOAD RAW DATA
def load_data() -> list[dict[str, Any]]:

    if not RAW_JSON.exists():
        raise FileNotFoundError(f"Raw MegaVul file not found: {RAW_JSON}")

    with RAW_JSON.open("r",encoding="utf-8",) as f:
        records = json.load(f)

    if not isinstance(records,list,):
        raise ValueError("MegaVul JSON must contain a list.")

    if not records:
        raise ValueError("MegaVul JSON contains no records.")

    return records


# TOKENIZER
def load_tokenizer():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.bos_token_id is None:
        raise ValueError("Tokenizer has no bos_token_id.")

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id.")

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id.")

    # We deliberately tokenize complete functions before applying
    # our own 4-chunk limit. Setting this very high suppresses the
    # generic Hugging Face >512 warning. It does NOT change the
    # produced tokens.
    tokenizer.model_max_length = 1_000_000_000

    return tokenizer


# BUILD LEAKAGE COMPONENTS
def build_components(records: list[dict[str, Any]],tokenizer,):

    uf = UnionFind(len(records))

    cve_owner: dict[str, int] = {}

    raw_code_owner: dict[str,int,] = {}

    semantic_owner: dict[str,int,] = {}

    raw_hashes: list[str] = []

    semantic_hashes: list[str] = []

    semantic_hash_labels = defaultdict(set)

    semantic_hash_sample_ids = defaultdict(list)

    print()
    print("Building leakage components...")

    print("Semantic configuration:")

    print(f"  model max length: {MODEL_MAX_LENGTH}")

    print(f"  content size: {CONTENT_SIZE}")

    print(f"  overlap: {OVERLAP}")

    print(f"  stride: {STRIDE}")

    print(f"  max chunks: {MAX_CHUNKS}")

    print(f"  visible capacity: {SEMANTIC_VISIBLE_CAPACITY}")

    for index, record in enumerate(records):
        code = get_code(record)

        raw_hash = code_hash(code)

        semantic_hash = semantic_input_hash(code,tokenizer,)

        raw_hashes.append(raw_hash)

        semantic_hashes.append(semantic_hash)

        label = int(bool(record.get("is_vul")))

        semantic_hash_labels[semantic_hash].add(label)

        semantic_hash_sample_ids[semantic_hash].append(index)

        # Same CVE -> same component
        cve = record.get("cve_id")

        if cve:
            if cve in cve_owner:
                uf.union(index,cve_owner[cve],)
            else:
                cve_owner[cve] = index

        # Same exact raw selected source -> same component

        if raw_hash in raw_code_owner:
            uf.union(index,raw_code_owner[raw_hash],)
        else:
            raw_code_owner[raw_hash] = index

        # Same semantic model-visible input -> same component
        if semantic_hash in semantic_owner:
            uf.union(index,semantic_owner[semantic_hash],)
        else:
            semantic_owner[semantic_hash] = index
       
        # Progress
        if (index + 1) % 1000 == 0:
            print(f"  processed {index + 1:,}/{len(records):,}")

    # MATERIALIZE COMPONENTS
    components = defaultdict(list)

    for index in range(len(records)):
        root = uf.find(index)

        components[root].append(index)

 
    # SEMANTIC DUPLICATE STATISTICS
    duplicate_semantic_hashes = {
        semantic_hash: sample_ids
        for (semantic_hash,sample_ids,) in semantic_hash_sample_ids.items()
        if len(sample_ids) > 1
    }

    conflicting_semantic_hashes = {
        semantic_hash: sample_ids
        for (semantic_hash,sample_ids,) in semantic_hash_sample_ids.items()
        if (len(semantic_hash_labels[semantic_hash]) > 1)
    }

    print()
    print("-- COMPONENT INPUT SUMMARY --")

    print(f"Unique CVEs: {len(cve_owner):,}")

    print(f"Unique raw-code hashes: {len(set(raw_hashes)):,}")

    print(f"Unique semantic-input hashes: {len(set(semantic_hashes)):,}")

    print(f"Duplicate semantic-input hashes: {len(duplicate_semantic_hashes):,}")

    print(f"Conflicting-label semantic hashes: {len(conflicting_semantic_hashes):,}")

    print(f"Final transitive leakage components: {len(components):,}")

    if conflicting_semantic_hashes:
        print()
        print("NOTE:")

        print("Identical semantic model inputs with different labels exist.")

        print("They are retained, but forced into the same leakage component.")

        print("This is label ambiguity for the semantic branch, not cross-split leakage.")

        print()
        print("Conflicting semantic groups:")

        for (semantic_hash,sample_ids,) in conflicting_semantic_hashes.items():
            labels = sorted(
                {
                    int(bool(records[sample_id].get("is_vul")))
                    for sample_id in sample_ids
                }
            )

            print(f"  hash={semantic_hash}")

            print(f"  sample_ids={sample_ids}")

            print(f"  labels={labels}")

    diagnostics = {
        "unique_cves": len(cve_owner),
        "unique_raw_code_hashes": len(set(raw_hashes)),
        "unique_semantic_input_hashes": len(set(semantic_hashes)),
        "duplicate_semantic_input_hashes": len(duplicate_semantic_hashes),
        "conflicting_label_semantic_hashes": len(conflicting_semantic_hashes),
        "leakage_components": len(components),
    }

    return (
        components,
        raw_hashes,
        semantic_hashes,
        diagnostics,
    )

# COMPONENT STATISTICS
def component_statistics(components,records,):

    stats = []
    for (root,indices,) in components.items():
        vulnerable = sum(1 for index in indices if bool(records[index].get("is_vul")))

        non_vulnerable = len(indices) - vulnerable

        stats.append(
            {
                "root": root,
                "indices": indices,
                "total": len(indices),
                "vulnerable": vulnerable,
                "non_vulnerable": non_vulnerable,
            }
        )

    return stats

# SPLIT OBJECTIVE
def objective(totals,vulnerable,target_totals,target_vulnerable,):

    total_error = 0.0

    vulnerable_error = 0.0

    for split in SPLITS:
        total_error += ((totals[split] - target_totals[split]) / target_totals[split]) ** 2

        vulnerable_error += ((vulnerable[split] - target_vulnerable[split]) / target_vulnerable[split]) ** 2

    return 0.55 * total_error + 0.45 * vulnerable_error

# SPLIT ASSIGNMENT
def create_split(component_stats,total_records,total_vulnerable,):

    target_totals = {
        split: total_records * fraction
        for (split,fraction,) in SPLITS.items()
    }

    target_vulnerable = {
        split: total_vulnerable * fraction
        for (split,fraction,) in SPLITS.items()
    }

    best_assignment = None

    best_score = None

    rng = random.Random(SEED)

    # DETERMINISTIC RESTARTS
    for _restart in range(30):
        components = list(component_stats)

        # Shuffle first so components with equal size can use a
        # deterministic randomized tie order.
        rng.shuffle(components)

        # Largest constrained components first.
        components.sort(key=lambda component: component["total"], reverse=True,)

        assignment = {}

        totals = {split: 0 for split in SPLITS}

        vulnerable = {split: 0 for split in SPLITS}

        for component in components:
            best_split = None

            best_candidate_score = None

            for split in SPLITS:
                new_totals = totals.copy()

                new_vulnerable = vulnerable.copy()

                new_totals[split] += component["total"]

                new_vulnerable[split] += component["vulnerable"]

                score = objective(
                    new_totals,
                    new_vulnerable,
                    target_totals,
                    target_vulnerable,
                )

                if best_candidate_score is None or score < best_candidate_score:
                    best_candidate_score = score

                    best_split = split

            if best_split is None:
                raise RuntimeError("Unable to choose a split.")

            assignment[component["root"]] = best_split

            totals[best_split] += component["total"]

            vulnerable[best_split] += component["vulnerable"]

        final_score = objective(
            totals,
            vulnerable,
            target_totals,
            target_vulnerable,
        )

        if best_score is None or final_score < best_score:
            best_score = final_score

            best_assignment = assignment.copy()

    if best_assignment is None:
        raise RuntimeError("Split optimization failed.")

    return (
        best_assignment,
        target_totals,
        target_vulnerable,
        best_score,
    )


  
# SPLIT SUMMARY
def build_summary(records,components,assignment,):

    summary = {
        split: {
            "records": 0,
            "vulnerable": 0,
            "non_vulnerable": 0,
            "components": set(),
            "cves": set(),
        }
        for split in SPLITS
    }

    for (root,indices,) in components.items():
        split = assignment[root]

        summary[split]["components"].add(root)

        for index in indices:
            record = records[index]

            summary[split]["records"] += 1

            cve = record.get("cve_id")

            if cve:
                summary[split]["cves"].add(cve)

            if bool(record.get("is_vul")):
                summary[split]["vulnerable"] += 1

            else:
                summary[split]["non_vulnerable"] += 1

    return summary


def print_summary(summary,) -> None:

    print()
    print("-- FINAL SPLIT SUMMARY --")

    for split in SPLITS:
        data = summary[split]

        print()
        print(split.upper())

        print("  Components:",len(data["components"]),)

        print("  CVEs:",len(data["cves"]),)

        print("  Records:",data["records"],)

        print("  Vulnerable:",data["vulnerable"],)

        print("  Non-vulnerable:",data["non_vulnerable"],)

        rate = 100 * data["vulnerable"] / data["records"]

        print("  Vulnerability rate:",f"{rate:.2f}%",)

# LEAKAGE VERIFICATION
def verify_leakage(records,raw_hashes,semantic_hashes,assignment,components,):

    split_cves = defaultdict(set)

    split_raw_hashes = defaultdict(set)

    split_semantic_hashes = defaultdict(set)

    split_components = defaultdict(set)

    for (root,indices,) in components.items():
        split = assignment[root]

        split_components[split].add(root)

        for index in indices:
            record = records[index]

            cve = record.get("cve_id")

            if cve:
                split_cves[split].add(cve)

            split_raw_hashes[split].add(raw_hashes[index])

            split_semantic_hashes[split].add(semantic_hashes[index])

    print()
    print("-- LEAKAGE VERIFICATION --")

    comparisons = {}

    for (split_a,split_b,) in [("train","validation",),("train","test",),("validation","test",),]:
        cve_overlap = split_cves[split_a] & split_cves[split_b]

        raw_hash_overlap = split_raw_hashes[split_a] & split_raw_hashes[split_b]

        semantic_hash_overlap = (
            split_semantic_hashes[split_a] & split_semantic_hashes[split_b]
        )

        component_overlap = split_components[split_a] & split_components[split_b]

        comparison_name = f"{split_a}_vs_{split_b}"

        comparisons[comparison_name] = {
            "cve_overlap": len(cve_overlap),
            "raw_code_hash_overlap": len(raw_hash_overlap),
            "semantic_input_hash_overlap": len(semantic_hash_overlap),
            "component_overlap": len(component_overlap),
        }

        print()
        print(f"{split_a} vs {split_b}")

        print("  CVE overlap:",len(cve_overlap),)

        print("  Raw code-hash overlap:",len(raw_hash_overlap),)

        print("  Semantic-input overlap:",len(semantic_hash_overlap),)

        print("  Component overlap:",len(component_overlap),)

        if cve_overlap:
            raise RuntimeError(f"CVE leakage detected: {split_a} vs {split_b}")

        if raw_hash_overlap:
            raise RuntimeError(f"Raw code leakage detected: {split_a} vs {split_b}")

        if semantic_hash_overlap:
            raise RuntimeError(
                f"Semantic model-input leakage detected: {split_a} vs {split_b}"
            )

        if component_overlap:
            raise RuntimeError(f"Component leakage detected: {split_a} vs {split_b}")

    print()
    print("LEAKAGE CHECK: PASSED")

    return comparisons

# WRITE MANIFEST AND SPLIT FILES
def write_files(records,raw_hashes,semantic_hashes,components,assignment,):

    MANIFEST.parent.mkdir(parents=True,exist_ok=True,)

    SPLIT_DIR.mkdir(parents=True,exist_ok=True,)

    fields = [
        "sample_id",
        "split",
        "leakage_component",
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
        "semantic_input_hash",
    ]

    rows = []

    for (root,indices,) in components.items():
        split = assignment[root]

        component_name = f"LC_{root:06d}"

        for index in indices:
            record = records[index]

            is_vulnerable = bool(record.get("is_vul"))

            if is_vulnerable:
                code_version = "before"

                graph_path = record.get("func_graph_path_before")

            else:
                code_version = "clean"

                graph_path = record.get("func_graph_path")

            if not graph_path:
                raise ValueError(f"Missing graph path for sample {index}.")

            cwe_ids = record.get("cwe_ids",[],)

            if cwe_ids is None:
                cwe_ids = []

            row = {
                "sample_id": index,
                "split": split,
                "leakage_component": component_name,
                "cve_id": record.get("cve_id"),
                "repo_name": record.get("repo_name"),
                "commit_hash": record.get("commit_hash"),
                "parent_commit_hash": record.get("parent_commit_hash"),
                "file_path": record.get("file_path"),
                "func_name": record.get("func_name"),
                "is_vul": int(is_vulnerable),
                "cwe_ids": "|".join(str(cwe_id) for cwe_id in cwe_ids),
                "code_version": code_version,
                "graph_path": graph_path,
                "diff_line_info": json.dumps(
                    record.get("diff_line_info"),
                    ensure_ascii=False,
                ),
                "code_hash": raw_hashes[index],
                "semantic_input_hash": semantic_hashes[index],
            }

            rows.append(row)

    # Stable sample ordering.
    rows.sort(key=lambda row: int(row["sample_id"]))

 
    # FULL MANIFEST
    with MANIFEST.open("w",encoding="utf-8",newline="",) as f:
        writer = csv.DictWriter(f,fieldnames=fields,)

        writer.writeheader()

        writer.writerows(rows)

 
    # INDIVIDUAL SPLITS
    for split in SPLITS:
        split_path = SPLIT_DIR / f"{split}.csv"

        split_rows = [row for row in rows if row["split"] == split]

        with split_path.open("w",encoding="utf-8",newline="",) as f:
            writer = csv.DictWriter(f,fieldnames=fields,)
            writer.writeheader()

            writer.writerows(split_rows)

    print()
    print("Created:",MANIFEST,)

    print("Created split files in:",SPLIT_DIR,)


  
# WRITE LEAKAGE REPORT
def write_leakage_report(diagnostics,comparisons,summary,optimization_score,):

    serializable_summary = {}

    for split in SPLITS:
        data = summary[split]

        serializable_summary[split] = {
            "records": int(data["records"]),
            "vulnerable": int(data["vulnerable"]),
            "non_vulnerable": int(data["non_vulnerable"]),
            "components": len(data["components"]),
            "cves": len(data["cves"]),
            "vulnerability_rate": (data["vulnerable"] / data["records"]),
        }

    report = {
        "seed": SEED,
        "model_name": MODEL_NAME,
        "semantic_preprocessing": {
            "model_max_length": MODEL_MAX_LENGTH,
            "special_tokens": SPECIAL_TOKENS,
            "content_size": CONTENT_SIZE,
            "overlap": OVERLAP,
            "stride": STRIDE,
            "max_chunks": MAX_CHUNKS,
            "visible_source_token_capacity": SEMANTIC_VISIBLE_CAPACITY,
            "comments_removed": True,
        },
        "grouping_rules": [
            "same_cve",
            "same_exact_selected_raw_code",
            "same_exact_semantic_model_input",
        ],
        "diagnostics": diagnostics,
        "optimization_score": float(optimization_score),
        "split_summary": serializable_summary,
        "pairwise_leakage": comparisons,
        "passed": True,
    }

    LEAKAGE_REPORT.parent.mkdir(parents=True,exist_ok=True,)

    with LEAKAGE_REPORT.open("w",encoding="utf-8",) as f:
        json.dump( report,f,indent=2,)

    print("Created:",LEAKAGE_REPORT,)

# MAIN
def main() -> None:

    print("-- BUILDING FINAL LEAKAGE-SAFE SPLIT --")

    print()
    print("Grouping rules:")

    print("  1. Same CVE")

    print("  2. Same exact selected raw source")

    print("  3. Same exact semantic model input")

 
    # LOAD
    records = load_data()

    print()
    print("Records:",f"{len(records):,}",)

    tokenizer = load_tokenizer()
    # COMPONENTS
    (components,raw_hashes,semantic_hashes,diagnostics,) = build_components(records,tokenizer,)

    # STATS
    stats = component_statistics(components,records,)

    total_records = len(records)

    total_vulnerable = sum(component["vulnerable"] for component in stats)

    print()
    print("Total vulnerable:",f"{total_vulnerable:,}",)

    print("Total non-vulnerable:",f"{total_records - total_vulnerable:,}",)

 
    # ASSIGN COMPONENTS
    (assignment,_target_totals,_target_vulnerable,score,) = create_split(stats,total_records,total_vulnerable,)

    print()
    print("Optimization score:",score,)

 
    # SUMMARY
    summary = build_summary(records,components,assignment,)

    print_summary(summary)

    # VERIFY BEFORE WRITING
    comparisons = verify_leakage(records,raw_hashes,semantic_hashes,assignment,components,)

    # WRITE
    write_files(records,raw_hashes,semantic_hashes,components,assignment,)

    write_leakage_report(diagnostics,comparisons,summary,score,)

    print()
    print("--")

    print("FINAL SPLIT CREATION COMPLETE")

    print("CVE leakage: 0")

    print("Raw code leakage: 0")

    print("Semantic model-input leakage: 0")

    print("Component leakage: 0")

    print("--")


if __name__ == "__main__":
    main()
