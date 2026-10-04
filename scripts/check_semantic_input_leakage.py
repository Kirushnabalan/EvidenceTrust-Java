from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer



# PROJECT ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT),)

from src.semantic.dataset import MegaVulSemanticDataset

# CONFIGURATION
MODEL_NAME = "microsoft/codebert-base"

MODEL_INPUT_DIR = PROJECT_ROOT / "dataset" / "processed" / "model_inputs"

SPLIT_FILES = {
    "train": MODEL_INPUT_DIR / "train.jsonl",
    "validation": MODEL_INPUT_DIR / "validation.jsonl",
    "test": MODEL_INPUT_DIR / "test.jsonl",
}



# HASHING
def hash_model_input(item,) -> str:

    digest = hashlib.sha256()

    for key in ["input_ids","attention_mask","chunk_mask",]:
        tensor = item[key].detach().cpu().contiguous()

        digest.update(key.encode("utf-8"))

        digest.update(str(tensor.dtype).encode("utf-8"))

        digest.update(str(tuple(tensor.shape)).encode("utf-8"))

        digest.update(tensor.numpy().tobytes())

    return digest.hexdigest()



# MAIN
def main() -> None:

    print("-- SEMANTIC MODEL-INPUT LEAKAGE AUDIT --")

    print()

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # hash  [(split, sample_id, label)]
    occurrences = defaultdict(list)

    total_records = 0

    split_counts = {}

    # HASH EVERY MODEL INPUT

    for split_name, split_file in SPLIT_FILES.items():
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")

        print()

        print(f"Processing {split_name}:")

        print(split_file)

        dataset = MegaVulSemanticDataset(split_file,tokenizer,)

        split_counts[split_name] = len(dataset)

        for index in range(len(dataset)):
            item = dataset[index]

            model_hash = hash_model_input(item)

            sample_id = item.get("sample_id",index,)

            if hasattr(sample_id,"item",):
                sample_id = sample_id.item()

            label = item["labels"]

            if hasattr(label,"item",):
                label = label.item()

            occurrences[model_hash].append((split_name,str(sample_id),int(label),))

            total_records += 1

            if (index + 1) % 1000 == 0:
                print(f"  processed {index + 1:,}/{len(dataset):,}")

    # ANALYZE COLLISIONS
    duplicate_hashes = {model_hash: rows for model_hash, rows in occurrences.items() if len(rows) > 1}

    cross_split = {}

    conflicting_labels = {}

    for model_hash, rows in duplicate_hashes.items():
        splits = {row[0] for row in rows}

        labels = {row[2] for row in rows}

        if len(splits) > 1:
            cross_split[model_hash] = rows

        if len(labels) > 1:
            conflicting_labels[model_hash] = rows

    # REPORT
    print()
    print("-- AUDIT RESULTS --")
    print()

    print("Split counts:")

    for split_name in ["train","validation","test",]:
        print(f"  {split_name:10}: {split_counts[split_name]:,}")

    print()

    print(f"Total records: {total_records:,}")

    print(f"Unique semantic model inputs: {len(occurrences):,}")

    print(f"Duplicate model-input hashes: {len(duplicate_hashes):,}")

    print(f"Cross-split duplicate hashes: {len(cross_split):,}")

    print(f"Conflicting-label duplicate hashes: {len(conflicting_labels):,}")

    # CROSS-SPLIT DETAILS
    if cross_split:
        print()
        print("-- CROSS-SPLIT COLLISIONS --")

        for collision_index, (model_hash,rows,) in enumerate(cross_split.items(),start=1,):
            print()

            print(f"Collision {collision_index}")

            print("Hash:",model_hash,)

            for (split_name,sample_id,label,) in rows:
                print(f"  split={split_name:10} sample_id={sample_id} label={label}")

    # FINAL STATUS
    print()
    

    if cross_split:
        print("RESULT: CROSS-SPLIT SEMANTIC INPUT LEAKAGE DETECTED")
        print("Do NOT start final semantic training yet.")

    else:
        print("RESULT: PASS")
        print("No identical semantic model inputs occur across splits.")

    


if __name__ == "__main__":
    main()
