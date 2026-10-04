"""
Semantic dataset for EvidenceTrust-Java.

Each original Java function remains ONE training/evaluation sample.

Semantic preprocessing:
    1. Load original Java source.
    2. Remove Java comments while preserving layout.
    3. Tokenize the complete cleaned function.
    4. Divide long functions into overlapping CodeBERT chunks.
    5. Keep at most MAX_CHUNKS chunks.
    6. Record semantic evidence coverage metadata.

Canonical V1 configuration:
    CodeBERT maximum sequence length = 512
    Source tokens per chunk          = 510
    Overlap                          = 128
    Stride                           = 382
    Maximum chunks                   = 4

Important:
    - No dataset splitting happens here.
    - Existing train/validation/test splits remain unchanged.
    - Semantic model input is Java source code only.
    - Target is is_vul.
    - CWE, CVE, graph_path, diff_line_info, etc. are NOT used
      as semantic model inputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.data.java_comments import (remove_java_comments_preserve_layout,)
# CANONICAL SEMANTIC CONFIGURATION
MODEL_MAX_LENGTH = 512
OVERLAP = 128
MAX_CHUNKS = 4


REQUIRED_FIELDS = {"sample_id","split","is_vul","code",}

class MegaVulSemanticDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_chunks: int = MAX_CHUNKS,
        overlap: int = OVERLAP,
        limit: Optional[int] = None,
    ) -> None:

        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_chunks = int(max_chunks)
        self.overlap = int(overlap)

        # Input validation
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {self.jsonl_path}")

        if self.max_chunks <= 0:
            raise ValueError("max_chunks must be greater than zero.")

        if self.overlap < 0:
            raise ValueError("overlap cannot be negative.")

         # CodeBERT / RoBERTa token IDs
 
        self.bos_token_id = self.tokenizer.bos_token_id

        self.eos_token_id = self.tokenizer.eos_token_id

        self.pad_token_id = self.tokenizer.pad_token_id

        if self.bos_token_id is None:
            raise ValueError("Tokenizer does not define bos_token_id.")

        if self.eos_token_id is None:
            raise ValueError("Tokenizer does not define eos_token_id.")

        if self.pad_token_id is None:
            raise ValueError("Tokenizer does not define pad_token_id.")

        self.special_tokens = 2

        self.content_size = MODEL_MAX_LENGTH - self.special_tokens

        if self.content_size <= 0:
            raise ValueError("MODEL_MAX_LENGTH is too small.")

        if self.overlap >= self.content_size:
            raise ValueError("overlap must be smaller than content_size.")

        self.stride = self.content_size - self.overlap

         # Load records
 
        self.records: list[Dict[str, Any]] = []

        self._load_records(limit=limit)

        self._validate_records()

    # RECORD LOADING
    def _load_records(self,limit: Optional[int],) -> None:

        with self.jsonl_path.open("r",encoding="utf-8",) as f:
            for line_number, line in enumerate(f,start=1,):
                if limit is not None and len(self.records) >= limit:
                    break

                line = line.strip()

                if not line:
                    continue
                try:
                    record = json.loads(line)

                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_number} in {self.jsonl_path}") from exc

                if not isinstance(record,dict,):
                    raise ValueError(f"Expected JSON object at line {line_number}.")

                self.records.append(record)

        if not self.records:
            raise ValueError(f"No records loaded from {self.jsonl_path}")

    
    # RECORD VALIDATION
    def _validate_records(self,) -> None:
        for index, record in enumerate(self.records):
            missing = REQUIRED_FIELDS - set(record.keys())

            if missing:
                raise ValueError(
                    f"Record index {index} "
                    f"is missing required fields: "
                    f"{sorted(missing)}"
                )

            # sample_id
            try:
                int(record["sample_id"])

            except (TypeError,ValueError,) as exc:
                raise ValueError(
                    f"Invalid sample_id at "
                    f"record index {index}: "
                    f"{record['sample_id']!r}"
                ) from exc

            # split
            split = str(record["split"])

            if split not in {"train","validation","test",}:
                raise ValueError(f"Invalid split at record index {index}: {split!r}")

            # label
            try:
                label = int(record["is_vul"])

            except (TypeError,ValueError,) as exc:
                raise ValueError(f"Invalid is_vul at record index {index}: {record['is_vul']!r}") from exc

            if label not in {0,1,}:
                raise ValueError(f"is_vul must be 0 or 1; got {label}")
            # code
            code = record["code"]

            if not isinstance(code,str,):
                raise ValueError(f"Code must be a string at record index {index}.")

            if not code.strip():
                raise ValueError(f"Empty source code for sample_id={record['sample_id']}")

    
    # SEMANTIC MODEL CODE PREPROCESSING
    @staticmethod
    def _prepare_model_code(raw_code: str,) -> str:
        model_code = remove_java_comments_preserve_layout(raw_code)

        if len(model_code) != len(raw_code):
            raise RuntimeError("Comment preprocessing changed source length.")

        if model_code.count("\n") != raw_code.count("\n"):
            raise RuntimeError("Comment preprocessing changed source line count.")

        # A function theoretically containing only comments
        # should not silently enter the model.
        if not model_code.strip():
            raise ValueError("Source became empty after comment removal.")

        return model_code
    
    # FULL FUNCTION TOKENIZATION
    def _tokenize_full_code(self,code: str,) -> list[int]:

        tokens = self.tokenizer.tokenize(code)

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        if not isinstance(token_ids,list,):
            token_ids = list(token_ids)

        if not token_ids:
            raise ValueError("Tokenizer produced zero tokens.")

        return token_ids

    
    # REQUIRED CHUNK COUNT
    def _required_chunks(self,token_count: int,) -> int:

        if token_count <= 0:
            return 0

        if token_count <= self.content_size:
            return 1

        remaining = token_count - self.content_size

        return 1 + math.ceil(remaining / self.stride)

    
    # SINGLE CHUNK ENCODING
    def _encode_chunk(self,source_token_ids: list[int],) -> tuple[list[int],list[int],]:

        if len(source_token_ids) > self.content_size:
            raise ValueError(
                f"Source chunk contains "
                f"{len(source_token_ids)} tokens; "
                f"maximum is "
                f"{self.content_size}."
            )

        input_ids = [self.bos_token_id,*source_token_ids,self.eos_token_id,]

        sequence_length = len(input_ids)

        if sequence_length > MODEL_MAX_LENGTH:
            raise RuntimeError(
                f"Constructed sequence length "
                f"{sequence_length} exceeds "
                f"{MODEL_MAX_LENGTH}."
            )

        attention_mask = [1] * sequence_length

        padding_length = MODEL_MAX_LENGTH - sequence_length

        input_ids.extend([self.pad_token_id] * padding_length)

        attention_mask.extend([0] * padding_length)

        if len(input_ids) != MODEL_MAX_LENGTH:
            raise RuntimeError("input_ids length is not 512.")

        if len(attention_mask) != MODEL_MAX_LENGTH:
            raise RuntimeError("attention_mask length is not 512.")

        return (input_ids,attention_mask,)

    
    # FUNCTION CHUNKING
    def _create_chunks(self,token_ids: list[int],) -> tuple[list[list[int]],list[list[int]],list[int],int,]:

        if not token_ids:
            raise ValueError("Cannot chunk an empty token sequence.")

        raw_chunks: list[list[int]] = []

        start = 0

         # Create overlapping source-code windows
 
        while start < len(token_ids) and len(raw_chunks) < self.max_chunks:
            end = start + self.content_size

            chunk = token_ids[start:end]

            if not chunk:
                break

            raw_chunks.append(chunk)

            # Entire function is covered.
            if end >= len(token_ids):
                break

            start += self.stride

        if not raw_chunks:
            raise RuntimeError("Chunk creation produced zero chunks.")

        encoded_chunks: list[list[int]] = []

        attention_masks: list[list[int]] = []

         # Encode actual chunks
 
        for chunk in raw_chunks:
            (
                chunk_input_ids,
                chunk_attention_mask,
            ) = self._encode_chunk(chunk)

            encoded_chunks.append(chunk_input_ids)

            attention_masks.append(chunk_attention_mask)

        real_chunk_count = len(encoded_chunks)

         # Function-level chunk mask
 
        chunk_mask = [1] * real_chunk_count + [0] * (self.max_chunks - real_chunk_count)

         # Pad unused chunk slots
 
        while len(encoded_chunks) < self.max_chunks:
            encoded_chunks.append([self.pad_token_id] * MODEL_MAX_LENGTH)

            attention_masks.append([0] * MODEL_MAX_LENGTH)

         # Defensive checks
 
        if len(encoded_chunks) != self.max_chunks:
            raise RuntimeError("Incorrect encoded chunk count.")

        if len(attention_masks) != self.max_chunks:
            raise RuntimeError("Incorrect attention-mask count.")

        if len(chunk_mask) != self.max_chunks:
            raise RuntimeError("Incorrect chunk-mask count.")

        for row in encoded_chunks:
            if len(row) != MODEL_MAX_LENGTH:
                raise RuntimeError("Invalid input_ids row length.")

        for row in attention_masks:
            if len(row) != MODEL_MAX_LENGTH:
                raise RuntimeError("Invalid attention_mask row length.")

        return (
            encoded_chunks,
            attention_masks,
            chunk_mask,
            real_chunk_count,
        )

    
    # DATASET INTERFACE
    def __len__(self,) -> int:

        return len(self.records)

    def __getitem__(self,index: int,) -> Dict[str, Any]:

        record = self.records[index]

        sample_id = int(record["sample_id"])

        label = int(record["is_vul"])

         # ORIGINAL SOURCE
        raw_code = record["code"]
        # Comments are removed ONLY from model input.
        # Original JSONL/source remains untouched.
 
        model_code = self._prepare_model_code(raw_code)

         # Tokenize complete comment-cleaned function
 
        token_ids = self._tokenize_full_code(model_code)

        original_token_count = len(token_ids)

        required_chunks = self._required_chunks(original_token_count)

         # Create CodeBERT chunks
 
        (input_ids,attention_masks,chunk_mask,used_chunks,) = self._create_chunks(token_ids)

         # Unique token coverage
        coverage_capacity = self.content_size + (used_chunks - 1) * self.stride

        covered_token_count = min(original_token_count,coverage_capacity,)

        semantic_coverage = covered_token_count / original_token_count

        was_truncated = covered_token_count < original_token_count

        max_chunks_reached = required_chunks > self.max_chunks

         # Return tensors
        return {
            # MODEL INPUTS
            "input_ids": torch.tensor(input_ids,dtype=torch.long,),
            "attention_mask": torch.tensor(attention_masks,dtype=torch.long,),
            "chunk_mask": torch.tensor(chunk_mask,dtype=torch.float32,),
            # LABEL / IDENTIFIER
            "labels": torch.tensor(label,dtype=torch.long,),
            "sample_id": torch.tensor(sample_id,dtype=torch.long,),
            # SEMANTIC EVIDENCE METADATA
            "original_token_count": torch.tensor(original_token_count,dtype=torch.long,),
            "covered_token_count": torch.tensor(covered_token_count,dtype=torch.long,),
            "semantic_coverage": torch.tensor(semantic_coverage,dtype=torch.float32,),
            "used_chunks": torch.tensor(used_chunks,dtype=torch.long,),
            "required_chunks": torch.tensor(required_chunks,dtype=torch.long,),
            "was_truncated": torch.tensor(int(was_truncated),dtype=torch.long,),
            "max_chunks_reached": torch.tensor(int(max_chunks_reached),dtype=torch.long,),
        }

    
    # HELPERS
    def class_counts(self,) -> Dict[int, int]:

        counts = {0: 0,1: 0,}

        for record in self.records:
            label = int(record["is_vul"])

            counts[label] += 1

        return counts

    def split_name(self,) -> str:

        splits = {str(record["split"]) for record in self.records}

        if len(splits) != 1:
            raise ValueError(f"Expected exactly one split; found {sorted(splits)}")

        return next(iter(splits))

    def configuration(self,) -> Dict[str, int]:
        return {
            "model_max_length": MODEL_MAX_LENGTH,
            "special_tokens": self.special_tokens,
            "content_size": self.content_size,
            "overlap": self.overlap,
            "stride": self.stride,
            "max_chunks": self.max_chunks,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }
