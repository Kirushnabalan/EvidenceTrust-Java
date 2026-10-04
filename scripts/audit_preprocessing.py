import csv
import json
import re
from collections import Counter
from pathlib import Path
# PATHS

RAW_JSON = Path("dataset/raw/megavul/megavul.json")

MANIFEST = Path("dataset/processed/megavul_manifest.csv")

GRAPH_ROOT = Path("dataset/raw/megavul/megavul_graph")

REPORT_DIR = Path("dataset/processed/preprocessing_audit")

REPORT_CSV = REPORT_DIR / "record_quality.csv"

SUMMARY_JSON = REPORT_DIR / "summary.json"


# CWE
CWE_PATTERN = re.compile(r"^CWE-\d+$")

# BASIC HELPERS

def is_empty(value):
    # Return True if the value is missing or an empty string.
    return value is None or not isinstance(value, str) or not value.strip()


def load_json():
    #Load the MegaVul JSON file.
    with RAW_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def graph_path_exists(relative_path):
    #Check whether a referenced graph file exists.

    if not relative_path:
        return False

    return (GRAPH_ROOT / relative_path).is_file()

# CWE CLASSIFICATION

def classify_cwe_list(cwes):

    if not isinstance(cwes, list) or len(cwes) == 0:
        return "missing"

    statuses = []

    for cwe in cwes:
        if not isinstance(cwe, str):
            statuses.append("malformed")
            continue

        cwe = cwe.strip()

        if cwe == "CWE-Other":
            statuses.append("other")

        elif CWE_PATTERN.fullmatch(cwe):
            statuses.append("numeric")

        else:
            statuses.append("malformed")

    # If any malformed value exists,
    # classify the whole field as malformed.
    if "malformed" in statuses:
        return "malformed"

    # Numeric CWE takes priority if present.
    if "numeric" in statuses:
        return "numeric"

    # Otherwise CWE-Other.
    if "other" in statuses:
        return "other"

    return "missing"

# MODEL CODE
def get_model_code(record):
    """
    Select the code used for vulnerability detection.

    Vulnerable:
        func_before

    Non-vulnerable:
        func
    """

    if bool(record.get("is_vul")):
        return record.get("func_before")

    return record.get("func")



# LOCALIZATION CHECK
def has_nonempty_diff_lines(diff_info):

    #Check whether diff_line_info contains at least one added or deleted line.

    if not isinstance(diff_info, dict):
        return False

    deleted_lines = diff_info.get("deleted_lines", [])

    added_lines = diff_info.get("added_lines", [])

    deleted_ok = isinstance(deleted_lines, list) and len(deleted_lines) > 0

    added_ok = isinstance(added_lines, list) and len(added_lines) > 0

    return deleted_ok or added_ok

# MAIN
def main():

    print("Loading MegaVul...")

    records = load_json()

    if not isinstance(records, list):
        raise ValueError("Expected MegaVul JSON top-level value to be a list.")

    print(f"Records loaded: {len(records):,}")

    # COUNTERS
    counts = Counter()

    quality_rows = []

    code_lengths = []

    line_counts = []

    cwe_counts = Counter()

    graph_parse_errors = []

    
    # PROCESS EVERY RECORD
    for index, record in enumerate(records):
        # Record structure

        if not isinstance(record, dict):
            counts["malformed_record"] += 1

            quality_rows.append(
                {
                    "sample_id": index,
                    "is_vul": "",
                    "cve_id": "",
                    "repo_name": "",
                    "commit_hash": "",
                    "code_length": 0,
                    "line_count": 0,
                    "cwe_status": "missing",
                    "cwe_valid": 0,
                    "graph_exists": 0,
                    "graph_json_valid": 0,
                    "graph_nodes": 0,
                    "graph_edges": 0,
                    "has_diff_info": 0,
                    "semantic_usable": 0,
                    "structural_usable": 0,
                    "flags": "malformed_record",
                }
            )

            continue

        # Label

        is_vul = bool(record.get("is_vul"))

        # Select model code

        model_code = get_model_code(record)

        # REQUIRED FIELDS

        required_fields = [
            "cve_id",
            "repo_name",
            "commit_hash",
            "file_path",
            "func_name",
            "is_vul",
            "cwe_ids",
        ]

        missing_fields = [
            field
            for field in required_fields
            if (field not in record or record[field] is None)
        ]

        # CODE CHECK

        code_empty = is_empty(model_code)

        if isinstance(model_code, str):
            code_length = len(model_code)

            line_count = len(model_code.splitlines())

        else:
            code_length = 0

            line_count = 0

        if not code_empty:
            code_lengths.append(code_length)

            line_counts.append(line_count)

        # CWE CHECK

        cwes = record.get("cwe_ids")

        # IMPORTANT:
        # cwe_status is calculated BEFORE being used.
        cwe_status = classify_cwe_list(cwes)

        # CWE-Other is valid.
        cwe_valid = cwe_status in {"numeric", "other"}

        # Count CWE status here,
        # after cwe_status has been created.
        counts[f"cwe_{cwe_status}"] += 1

        if isinstance(cwes, list):
            for cwe in cwes:
                if isinstance(cwe, str):
                    cwe_counts[cwe.strip()] += 1

        # GRAPH CHECK

        if is_vul:
            graph_relative_path = record.get("func_graph_path_before")

        else:
            graph_relative_path = record.get("func_graph_path")

        graph_exists = graph_path_exists(graph_relative_path)

        graph_json_valid = False

        graph_nodes = 0

        graph_edges = 0

        if graph_exists:
            graph_path = GRAPH_ROOT / graph_relative_path

            try:
                with graph_path.open("r", encoding="utf-8") as f:
                    graph = json.load(f)

                if not isinstance(graph, dict):
                    graph_parse_errors.append(index)

                else:
                    nodes = graph.get("nodes")

                    edges = graph.get("edges")

                    if isinstance(nodes, list) and isinstance(edges, list):
                        graph_json_valid = True

                        graph_nodes = len(nodes)

                        graph_edges = len(edges)

                    else:
                        graph_parse_errors.append(index)

            except Exception:
                graph_parse_errors.append(index)

        # LOCALIZATION CHECK

        diff_info = record.get("diff_line_info")

        has_diff_info = has_nonempty_diff_lines(diff_info)

        # QUALITY FLAGS

        flags = []

        if missing_fields:
            flags.append("missing_required_field")

        if code_empty:
            flags.append("empty_model_code")

        if not cwe_valid:
            flags.append("invalid_cwe")

        if not graph_exists:
            flags.append("missing_graph")

        elif not graph_json_valid:
            flags.append("invalid_graph_json")

        if is_vul and not has_diff_info:
            flags.append("vulnerable_without_diff_info")

        # USABILITY

        usable_semantic = not code_empty and not missing_fields

        usable_structural = usable_semantic and graph_exists and graph_json_valid

        # GENERAL COUNTERS

        counts["total"] += 1

        if is_vul:
            counts["vulnerable"] += 1

        else:
            counts["non_vulnerable"] += 1

        if code_empty:
            counts["empty_model_code"] += 1

        if missing_fields:
            counts["missing_required_field"] += 1

        if not cwe_valid:
            counts["invalid_cwe"] += 1

        if not graph_exists:
            counts["missing_graph"] += 1

        if graph_exists and not graph_json_valid:
            counts["invalid_graph_json"] += 1

        if is_vul and not has_diff_info:
            counts["vulnerable_without_diff_info"] += 1

        if has_diff_info:
            counts["has_diff_info"] += 1

        if usable_semantic:
            counts["semantic_usable"] += 1

        if usable_structural:
            counts["structural_usable"] += 1

        # SAVE RECORD-LEVEL AUDIT

        quality_rows.append(
            {
                "sample_id": index,
                "is_vul": int(is_vul),
                "cve_id": record.get("cve_id", ""),
                "repo_name": record.get("repo_name", ""),
                "commit_hash": record.get("commit_hash", ""),
                "code_length": code_length,
                "line_count": line_count,
                "cwe_status": cwe_status,
                "cwe_valid": int(cwe_valid),
                "graph_exists": int(graph_exists),
                "graph_json_valid": int(graph_json_valid),
                "graph_nodes": graph_nodes,
                "graph_edges": graph_edges,
                "has_diff_info": int(has_diff_info),
                "semantic_usable": int(usable_semantic),
                "structural_usable": int(usable_structural),
                "flags": "|".join(flags),
            }
        )

    
    # TOTAL RECORD COUNT
    

    # The current MegaVul dataset should contain dictionaries
    # for every record. This keeps the count correct even if
    # malformed top-level records are encountered.
    if counts["malformed_record"]:
        counts["total"] += counts["malformed_record"]

    
    # CREATE OUTPUT DIRECTORY
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    
    # WRITE RECORD-LEVEL CSV
    if quality_rows:
        fields = list(quality_rows[0].keys())

        with REPORT_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)

            writer.writeheader()

            writer.writerows(quality_rows)

    
    # SAFE STATISTICS
    def safe_min(values):

        return min(values) if values else 0

    def safe_max(values):

        return max(values) if values else 0

    def safe_mean(values):

        return sum(values) / len(values) if values else 0

    
    # SUMMARY
    summary = {
        "records": counts["total"],
        "vulnerable": counts["vulnerable"],
        "non_vulnerable": counts["non_vulnerable"],
        "malformed_record": counts["malformed_record"],
        # Semantic / structural
        "semantic_usable": counts["semantic_usable"],
        "structural_usable": counts["structural_usable"],
        # Code
        "empty_model_code": counts["empty_model_code"],
        "missing_required_field": counts["missing_required_field"],
        # CWE
        "cwe_numeric": counts["cwe_numeric"],
        "cwe_other": counts["cwe_other"],
        "cwe_malformed": counts["cwe_malformed"],
        "cwe_missing": counts["cwe_missing"],
        "invalid_cwe": counts["invalid_cwe"],
        # Graph
        "missing_graph": counts["missing_graph"],
        "invalid_graph_json": counts["invalid_graph_json"],
        "graph_parse_errors": len(graph_parse_errors),
        # Localization
        "vulnerable_without_diff_info": counts["vulnerable_without_diff_info"],
        "has_diff_info": counts["has_diff_info"],
        # Code length
        "code_length": {
            "minimum": safe_min(code_lengths),
            "maximum": safe_max(code_lengths),
            "mean": safe_mean(code_lengths),
        },
        # Line count
        "line_count": {
            "minimum": safe_min(line_counts),
            "maximum": safe_max(line_counts),
            "mean": safe_mean(line_counts),
        },
        # CWE frequency
        "top_cwes": cwe_counts.most_common(20),
        # Graph errors
        "graph_parse_error_sample": graph_parse_errors[:20],
    }

    
    # WRITE JSON SUMMARY
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    
    # PRINT SUMMARY
    print("\n-- PREPROCESSING AUDIT --")

    print(f"{'total':35} {counts['total']:,}")

    print(f"{'vulnerable':35} {counts['vulnerable']:,}")

    print(f"{'non_vulnerable':35} {counts['non_vulnerable']:,}")

    print(f"{'malformed_record':35} {counts['malformed_record']:,}")

    print(f"{'semantic_usable':35} {counts['semantic_usable']:,}")

    print(f"{'structural_usable':35} {counts['structural_usable']:,}")

    print(f"{'empty_model_code':35} {counts['empty_model_code']:,}")

    print(f"{'missing_required_field':35} {counts['missing_required_field']:,}")

    print(f"{'cwe_numeric':35} {counts['cwe_numeric']:,}")

    print(f"{'cwe_other':35} {counts['cwe_other']:,}")

    print(f"{'cwe_malformed':35} {counts['cwe_malformed']:,}")

    print(f"{'cwe_missing':35} {counts['cwe_missing']:,}")

    print(f"{'invalid_cwe':35} {counts['invalid_cwe']:,}")

    print(f"{'missing_graph':35} {counts['missing_graph']:,}")

    print(f"{'invalid_graph_json':35} {counts['invalid_graph_json']:,}")

    print(
        f"{'vulnerable_without_diff_info':35} "
        f"{counts['vulnerable_without_diff_info']:,}"
    )

    print(f"{'has_diff_info':35} {counts['has_diff_info']:,}")

    
    # CODE SIZE
    

    print("\n-- CODE SIZE --")

    print("Minimum characters:", safe_min(code_lengths))

    print("Maximum characters:", safe_max(code_lengths))

    print("Mean characters:", f"{safe_mean(code_lengths):.2f}")

    print("\nMinimum lines:", safe_min(line_counts))

    print("Maximum lines:", safe_max(line_counts))

    print("Mean lines:", f"{safe_mean(line_counts):.2f}")

    
    # TOP CWE
    print("\n-- TOP CWE TYPES --")

    for cwe, count in cwe_counts.most_common(20):
        print(f"{cwe:12} {count:,}")

    
    # OUTPUT
    print("\n-- OUTPUT --")

    print(REPORT_CSV)

    print(SUMMARY_JSON)



# ENTRY POINT


if __name__ == "__main__":
    main()
