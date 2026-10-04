import csv
import json
from pathlib import Path

# Paths
RAW_JSON = Path("dataset/raw/megavul/megavul.json")
SPLIT_DIR = Path("dataset/splits")
OUTPUT_DIR = Path("dataset/processed/model_inputs")

# Split files
SPLITS = {
    "train": SPLIT_DIR / "train.csv",
    "validation": SPLIT_DIR / "validation.csv",
    "test": SPLIT_DIR / "test.csv",
}


def load_megavul():
    print("Loading MegaVul...")

    with RAW_JSON.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("MegaVul JSON must contain a list.")

    print(f"Loaded {len(records):,} records.")
    return records


def load_split(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_model_code(record):
    is_vul = bool(record.get("is_vul"))
    if is_vul:
        return record.get("func_before")
    return record.get("func")


def get_graph_path(record):
    is_vul = bool(record.get("is_vul"))
    if is_vul:
        return record.get("func_graph_path_before")
    return record.get("func_graph_path")


def prepare_record(raw_record, split_row):
    sample_id = int(split_row["sample_id"])
    is_vul = bool(int(split_row["is_vul"]))
    code = get_model_code(raw_record)
    graph_path = get_graph_path(raw_record)
    cwe_ids = raw_record.get("cwe_ids", [])
    diff_line_info = raw_record.get("diff_line_info")

    prepared = {
        # Identity
        "sample_id": sample_id,
        "split": split_row["split"],
        "leakage_component": split_row["leakage_component"],
        # Ground truth
        "is_vul": is_vul,
        "cwe_ids": cwe_ids,
        # Repository information
        "cve_id": raw_record.get("cve_id"),
        "repo_name": raw_record.get("repo_name"),
        "commit_hash": raw_record.get("commit_hash"),
        "parent_commit_hash": raw_record.get("parent_commit_hash"),
        "file_path": raw_record.get("file_path"),
        "func_name": raw_record.get("func_name"),
        # Semantic model input
        "code_version": split_row["code_version"],
        "code": code,
        # Structural model input
        "graph_path": graph_path,
        # Localization evidence
        "diff_line_info": diff_line_info,
        # Leakage audit information
        "code_hash": split_row["code_hash"],
    }

    return prepared


def validate_record(record):
    errors = []

    if not record["code"]:
        errors.append("missing_code")

    if not record["graph_path"]:
        errors.append("missing_graph_path")

    if not record["cwe_ids"]:
        errors.append("missing_cwe")

    if record["is_vul"]:
        if not record["diff_line_info"]:
            errors.append("vulnerable_missing_diff")
    else:
        # Non-vulnerable samples should not require vulnerability localization.
        pass

    return errors


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_records = load_megavul()
    all_sample_ids = set()
    total_errors = 0

    for split_name, split_path in SPLITS.items():
        print(f"\n {split_name.upper()} ")

        split_rows = load_split(split_path)
        prepared_records = []
        split_sample_ids = set()

        for row in split_rows:
            sample_id = int(row["sample_id"])

            # Check sample ID
            if sample_id < 0:
                raise ValueError(f"Invalid sample_id: {sample_id}")

            if sample_id >= len(raw_records):
                raise ValueError(f"sample_id {sample_id} is outside MegaVul.")

            if sample_id in split_sample_ids:
                raise ValueError(f"Duplicate sample_id {sample_id} in {split_name}.")

            split_sample_ids.add(sample_id)
            all_sample_ids.add(sample_id)
            raw_record = raw_records[sample_id]

            # Verify label consistency
            raw_label = bool(raw_record.get("is_vul"))
            csv_label = bool(int(row["is_vul"]))

            if raw_label != csv_label:
                raise ValueError(
                    f"Label mismatch for sample_id {sample_id}: "
                    f"raw={raw_label}, split={csv_label}"
                )

            # Prepare
            prepared = prepare_record(raw_record, row)

            # Validate
            errors = validate_record(prepared)
            if errors:
                total_errors += len(errors)
                print(f"WARNING sample_id={sample_id}: {errors}")

            prepared_records.append(prepared)

        output_path = OUTPUT_DIR / f"{split_name}.jsonl"
        write_jsonl(output_path, prepared_records)

        print(f"Records: {len(prepared_records):,}")
        print(f"Written: {output_path}")

    # Check sample coverage
    expected_ids = set(range(len(raw_records)))
    missing_ids = expected_ids - all_sample_ids
    extra_ids = all_sample_ids - expected_ids

    print("\n--COVERAGE CHECK--")
    print(f"MegaVul records: {len(raw_records):,}")
    print(f"IDs represented in splits: {len(all_sample_ids):,}")
    print(f"Missing IDs: {len(missing_ids):,}")
    print(f"Invalid extra IDs: {len(extra_ids):,}")

    if missing_ids:
        print("WARNING: some MegaVul records are not represented in the split files.")

    if extra_ids:
        raise ValueError("Split contains invalid sample IDs.")

    # Final summary
    print("\n-- PREPARATION COMPLETE --")
    print(f"Validation errors/warnings: {total_errors}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
