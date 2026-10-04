from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer
# Project setup

# Find the main project directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow Python to import modules from the project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.java_comments import remove_java_comments_preserve_layout

# Configuration

MODEL_NAME = "microsoft/codebert-base"

TRAIN_FILE = (PROJECT_ROOT / "dataset" / "processed" / "model_inputs" / "train.jsonl")
MODEL_MAX_LENGTH = 512


def calculate_percentile(values: list[int], percentile: float) -> float:
    #Calculate a percentile without using NumPy.
    if not values:
        return 0.0

    values = sorted(values)

    position = (len(values) - 1) * percentile

    lower_index = int(position)
    upper_index = min(lower_index + 1, len(values) - 1)

    fraction = position - lower_index

    lower_value = values[lower_index]
    upper_value = values[upper_index]

    return lower_value + (upper_value - lower_value) * fraction


def tokenize_full_code(tokenizer, code: str) -> list[int]:
    tokens = tokenizer.tokenize(code)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Ensure the result is always a Python list.
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)

    return token_ids


def main() -> None:
    #Run the token-length audit on the training dataset.

    # Load CodeBERT tokenizer

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # CodeBERT normally adds <s> and </s>.
    special_token_count = tokenizer.num_special_tokens_to_add(pair=False)

    print(f"Tokenizer: {MODEL_NAME}")
    print(f"Dataset: {TRAIN_FILE}")
    print(f"Special tokens added by model: {special_token_count}")
    print()

    # Store token lengths for all valid samples.
    source_token_lengths = []
    model_sequence_lengths = []

    total_records = 0
    empty_raw_code = 0
    empty_after_cleaning = 0

    # Read the training dataset
    with TRAIN_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            # Ignore blank lines in the JSONL file.
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON found at line {line_number}."
                ) from error

            total_records += 1

            raw_code = record.get("code")

            # Check that the sample contains valid Java code.
            if not isinstance(raw_code, str) or not raw_code.strip():
                empty_raw_code += 1
                continue

            # Apply the same cleaning used by the semantic model

            cleaned_code = remove_java_comments_preserve_layout(raw_code)

            if not cleaned_code.strip():
                empty_after_cleaning += 1
                continue

            # Tokenize the complete source code.
            token_ids = tokenize_full_code(
                tokenizer,
                cleaned_code,
            )

            if not token_ids:
                empty_after_cleaning += 1
                continue

            # Number of tokens from the actual source code.
            source_length = len(token_ids)

            # Number of tokens CodeBERT receives after adding
            # its special tokens.
            model_length = source_length + special_token_count

            source_token_lengths.append(source_length)
            model_sequence_lengths.append(model_length)

    # Validate the dataset

    if not model_sequence_lengths:
        raise RuntimeError("No valid records were found for the semantic model.")

    if empty_raw_code > 0:
        raise RuntimeError(f"Found {empty_raw_code:,} records with empty raw code.")

    if empty_after_cleaning > 0:
        raise RuntimeError(
            f"Found {empty_after_cleaning:,} records that became "
            "empty after comment removal."
        )

    source_token_lengths.sort()
    model_sequence_lengths.sort()

    measured_records = len(model_sequence_lengths)

    # Print audit results

    print("-- TOKEN LENGTH AUDIT --")
    print(f"Records in file: {total_records:,}")
    print(f"Records measured: {measured_records:,}")
    print(f"Empty raw code: {empty_raw_code:,}")
    print(f"Empty after comment cleaning: {empty_after_cleaning:,}")
    print()

    # Source-code token statistics

    print("-- SOURCE TOKEN STATISTICS --")
    print(f"Minimum: {min(source_token_lengths):,}")
    print(f"Maximum: {max(source_token_lengths):,}")
    print(f"Mean: {mean(source_token_lengths):,.2f}")
    print()

    # Model sequence statistics

    print("-- MODEL SEQUENCE STATISTICS --")
    print(
        f"Includes {special_token_count} special tokens "
        "added by CodeBERT."
    )
    print(f"Minimum: {min(model_sequence_lengths):,}")
    print(f"Maximum: {max(model_sequence_lengths):,}")
    print(f"Mean: {mean(model_sequence_lengths):,.2f}")
    print()

    # Token-length percentiles

    print("-- MODEL TOKEN PERCENTILES --")

    percentiles = [
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]

    for p in percentiles:
        value = calculate_percentile(model_sequence_lengths,p,)

        print(f"P{int(p * 100):02d}: {value:,.1f}")

    print()

    # Coverage at different maximum sequence lengths

    print("-- COVERAGE BY MAXIMUM LENGTH --")

    sequence_limits = [
        128,
        256,
        384,
        MODEL_MAX_LENGTH,
    ]

    for limit in sequence_limits:
        covered_records = sum(length <= limit for length in model_sequence_lengths)

        truncated_records = measured_records - covered_records

        coverage_percent = (covered_records / measured_records) * 100

        truncation_percent = (truncated_records / measured_records) * 100

        print(
            f"{limit:>3} tokens | "
            f"covered={covered_records:>6,} "
            f"({coverage_percent:6.2f}%) | "
            f"truncated={truncated_records:>6,} "
            f"({truncation_percent:6.2f}%)"
        )

    print()

    # Count very long source-code samples

    print("-- LONG-CODE COUNTS --")

    for limit in [256, 384, 512]:
        long_code_count = sum(length > limit for length in model_sequence_lengths)

        print(f"> {limit} tokens: " f"{long_code_count:,}")

    print()
    print("AUDIT COMPLETE")


if __name__ == "__main__":
    main()