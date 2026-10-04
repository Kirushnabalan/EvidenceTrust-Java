from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer

# PROJECT ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT),)


from src.data.java_comments import (remove_java_comments_preserve_layout,)

# CONFIGURATION
MODEL_NAME = "microsoft/codebert-base"

TRAIN_FILE = PROJECT_ROOT / "dataset" / "processed" / "model_inputs" / "train.jsonl"

MODEL_MAX_LENGTH = 512
SPECIAL_TOKENS = 2
OVERLAP = 128

CANDIDATE_MAX_CHUNKS = [
    1,
    2,
    4,
    8,
    16,
]


def percentile(values,p,):
    #Calculate percentile without requiring NumPy.

    values = sorted(values)

    if not values:
        return 0.0

    position = (len(values) - 1) * p

    lower = int(position)

    upper = min(lower + 1,len(values) - 1,)

    fraction = position - lower

    return values[lower] + (values[upper] - values[lower]) * fraction


def tokenize_source_code(tokenizer,code: str,) -> list[int]:

    tokens = tokenizer.tokenize(code)

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    if not isinstance(token_ids,list,):
        token_ids = list(token_ids)

    return token_ids


def required_chunks(token_count: int,content_size: int,stride: int,) -> int:

    #Number of overlapping chunks required to cover all source-code tokens.

    if token_count <= 0:
        return 0

    if token_count <= content_size:
        return 1

    remaining = token_count - content_size

    return 1 + math.ceil(remaining / stride)


def covered_tokens(token_count: int,content_size: int,stride: int,max_chunks: int,) -> int:
    
    #Number of UNIQUE source-code tokens that can be represented using at most max_chunks overlapping chunks.
    if token_count <= 0:
        return 0

    capacity = content_size + (max_chunks - 1) * stride

    return min(token_count,capacity,)


def main():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


    content_size = MODEL_MAX_LENGTH - SPECIAL_TOKENS

    stride = content_size - OVERLAP

    if content_size <= 0:
        raise ValueError("Invalid content_size.")

    if stride <= 0:
        raise ValueError("Overlap must be smaller than content size.")

    # CONFIGURATION

    print()

    print("-- CHUNK CONFIGURATION --")

    print("Model:",MODEL_NAME,)

    print("Model maximum length:",MODEL_MAX_LENGTH,)

    print("Special tokens:",SPECIAL_TOKENS,)

    print("Source-code tokens per chunk:",content_size,)

    print("Overlap:",OVERLAP,)

    print("Stride:",stride,)

    print()

    token_lengths = []
    chunk_counts = []

    total_records = 0
    empty_raw = 0
    empty_after_cleaning = 0

    # DATASET PROCESSING

    with TRAIN_FILE.open("r",encoding="utf-8",) as f:
        for line_number, line in enumerate(f,start=1,):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}") from exc

            total_records += 1

            raw_code = record.get("code")

            if (not isinstance(raw_code,str,) or not raw_code.strip()):
                empty_raw += 1
                continue

            # Same preprocessing used in semantic dataset

            model_code = remove_java_comments_preserve_layout(raw_code)

            if not model_code.strip():
                empty_after_cleaning += 1
                continue

            token_ids = tokenize_source_code(tokenizer,model_code,)

            if not token_ids:
                empty_after_cleaning += 1
                continue

            token_count = len(token_ids)

            chunks = required_chunks(token_count,content_size,stride,)

            token_lengths.append(token_count)

            chunk_counts.append(chunks)

    # VALIDATION

    if not chunk_counts:
        raise RuntimeError("No valid semantic-model functions were found.")

    if empty_raw > 0:
        raise RuntimeError(f"Found {empty_raw:,} records with empty raw code.")

    if empty_after_cleaning > 0:
        raise RuntimeError(
            f"Found {empty_after_cleaning:,} records "
            f"that became empty after comment cleaning."
        )

    total = len(chunk_counts)

    # DATASET SUMMARY

    print("-- DATASET --")

    print(f"Records in file: {total_records:,}")

    print(f"Functions measured: {total:,}")

    print(f"Empty raw code: {empty_raw:,}")

    print(f"Empty after comment cleaning: {empty_after_cleaning:,}")

    print()

    # REQUIRED CHUNK DISTRIBUTION

    print("-- REQUIRED CHUNKS --")

    print("Minimum:",min(chunk_counts),)

    print("Maximum:",max(chunk_counts),)

    print("Mean:",f"{mean(chunk_counts):.2f}",)

    print()

    print("P50:",f"{percentile(chunk_counts, 0.50):.1f}",)

    print("P75:",f"{percentile(chunk_counts, 0.75):.1f}",)

    print("P90:",f"{percentile(chunk_counts, 0.90):.1f}",)

    print("P95:",f"{percentile(chunk_counts, 0.95):.1f}",)

    print("P99:",f"{percentile(chunk_counts, 0.99):.1f}",)

    print()

    # FUNCTIONS REQUIRING MORE THAN N CHUNKS

    print("-- FUNCTIONS REQUIRING MORE THAN N CHUNKS --")

    for limit in CANDIDATE_MAX_CHUNKS:
        count = sum(chunks > limit for chunks in chunk_counts)

        percentage = count / total * 100

        print(f"> {limit:2} chunks: {count:6,} ({percentage:6.2f}%)")

    print()

    # MAX-CHUNK COVERAGE

    print("-- MAX-CHUNK COVERAGE --")

    for max_chunks in CANDIDATE_MAX_CHUNKS:
        coverage_values = []

        fully_covered = 0

        total_source_tokens = 0
        total_covered_tokens = 0

        for token_count in token_lengths:
            covered = covered_tokens(token_count,content_size,stride,max_chunks,)

            sample_coverage = covered / token_count

            coverage_values.append(sample_coverage)

            total_source_tokens += token_count

            total_covered_tokens += covered

            if covered >= token_count:
                fully_covered += 1

        mean_sample_coverage = mean(coverage_values) * 100

        full_percentage = fully_covered / total * 100

        global_token_coverage = total_covered_tokens / total_source_tokens * 100

        print(
            f"{max_chunks:2} chunks | "
            f"fully covered="
            f"{fully_covered:6,} "
            f"({full_percentage:6.2f}%) | "
            f"mean function coverage="
            f"{mean_sample_coverage:6.2f}% | "
            f"global token coverage="
            f"{global_token_coverage:6.2f}%"
        )

    print()

    #CONFIGURATION

    chosen_max_chunks = 4

    chosen_full = sum(chunks <= chosen_max_chunks for chunks in chunk_counts)

    chosen_percentage = chosen_full / total * 100

    print("-- CURRENT V1 CONFIGURATION --")

    print("MAX_CHUNKS:",chosen_max_chunks,)

    print("Functions fully covered:",f"{chosen_full:,}",)

    print("Fully covered percentage:",f"{chosen_percentage:.2f}%",)

    print()

    print("AUDIT COMPLETE")


if __name__ == "__main__":
    main()
