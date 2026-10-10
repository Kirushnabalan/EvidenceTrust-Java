from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel


DEFAULT_MODEL_NAME = "microsoft/codebert-base"


class SemanticVulnerabilityModel(nn.Module):

    def __init__(self,model_name: str = DEFAULT_MODEL_NAME,num_labels: int = 2,dropout: float = 0.1,) -> None:

        super().__init__()

        if num_labels != 2:
            raise ValueError(
                "EvidenceTrust semantic V1 expects "
                "binary classification with num_labels=2."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.model_name = model_name
        self.num_labels = num_labels

        # CodeBERT encoder
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = getattr(self.encoder.config,"hidden_size",None,)

        if hidden_size is None:
            raise ValueError("Could not determine encoder hidden_size.")

        self.hidden_size = int(hidden_size)

        # Function-level classification head
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            self.hidden_size,
            self.num_labels,
        )

    # CHUNK ENCODING
    def _encode_real_chunks(self,input_ids: torch.Tensor,attention_mask: torch.Tensor,chunk_mask: torch.Tensor,) -> torch.Tensor:

        if input_ids.ndim != 3:
            raise ValueError("input_ids must have shape [batch, chunks, sequence_length].")

        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")

        batch_size, max_chunks, sequence_length = input_ids.shape

        if chunk_mask.shape != (batch_size,max_chunks,):
            raise ValueError("chunk_mask must have shape [batch, chunks].")

        # Flatten function/chunk dimensions

        flat_input_ids = input_ids.reshape(batch_size * max_chunks,sequence_length,)

        flat_attention_mask = attention_mask.reshape(batch_size * max_chunks,sequence_length,)

        flat_chunk_mask = chunk_mask.reshape(batch_size * max_chunks)

        # Real chunk indices.
        real_chunk_indices = flat_chunk_mask > 0

        real_chunk_count = int(real_chunk_indices.sum().item())

        if real_chunk_count == 0:
            raise ValueError("Batch contains no real chunks.")

        # Select only real chunks
        real_input_ids = flat_input_ids[real_chunk_indices]

        real_attention_mask = flat_attention_mask[real_chunk_indices]

        # CodeBERT
        encoder_outputs = self.encoder(input_ids=real_input_ids,attention_mask=real_attention_mask,)

        last_hidden_state = encoder_outputs.last_hidden_state

        if last_hidden_state.ndim != 3:
            raise RuntimeError("Unexpected encoder output shape.")


        real_chunk_embeddings = last_hidden_state[:, 0, :]

        # Reconstruct padded function/chunk tensor

        flat_chunk_embeddings = torch.zeros(
            (
                batch_size * max_chunks,
                self.hidden_size,
            ),
            dtype=real_chunk_embeddings.dtype,
            device=real_chunk_embeddings.device,
        )

        flat_chunk_embeddings[real_chunk_indices] = real_chunk_embeddings

        chunk_embeddings = flat_chunk_embeddings.reshape(
            batch_size,
            max_chunks,
            self.hidden_size,
        )

        return chunk_embeddings

    # FUNCTION AGGREGATION

    def _aggregate_chunks(self,chunk_embeddings: torch.Tensor,chunk_mask: torch.Tensor,) -> torch.Tensor:

        if chunk_embeddings.ndim != 3:
            raise ValueError(
                "chunk_embeddings must have shape [batch, chunks, hidden_size]."
            )

        if chunk_mask.ndim != 2:
            raise ValueError("chunk_mask must have shape [batch, chunks].")

        if chunk_embeddings.shape[0] != chunk_mask.shape[0]:
            raise ValueError("Batch dimensions do not match.")

        if chunk_embeddings.shape[1] != chunk_mask.shape[1]:
            raise ValueError("Chunk dimensions do not match.")

        # [batch, chunks, 1]
        expanded_mask = chunk_mask.to(
            dtype=chunk_embeddings.dtype,
            device=chunk_embeddings.device,
        ).unsqueeze(-1)

        # Sum representations from real chunks.
        summed_embeddings = (chunk_embeddings * expanded_mask).sum(dim=1)

        # Number of real chunks per function.
        chunk_counts = expanded_mask.sum(dim=1)

        if torch.any(chunk_counts <= 0):
            raise ValueError("At least one function has zero real chunks.")

        function_embeddings = summed_embeddings / chunk_counts

        return function_embeddings


    # FORWARD
    def forward(self,input_ids: torch.Tensor,attention_mask: torch.Tensor,chunk_mask: torch.Tensor,labels: Optional[torch.Tensor] = None,) -> Dict[str, torch.Tensor]:
        # Encode source-code chunks
        chunk_embeddings = self._encode_real_chunks(
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_mask=chunk_mask,
        )

        # Aggregate into one function representation
        function_embeddings = self._aggregate_chunks(
            chunk_embeddings=chunk_embeddings,
            chunk_mask=chunk_mask,
        )

        # Classification
        dropped = self.dropout(function_embeddings)

        logits = self.classifier(dropped)

        output: Dict[str, torch.Tensor] = {
            "logits": logits,
            "function_embeddings": function_embeddings,
            "chunk_embeddings": chunk_embeddings,
        }
        # Optional smoke-test loss
        if labels is not None:
            if labels.ndim != 1:
                raise ValueError("labels must have shape [batch].")

            if labels.shape[0] != logits.shape[0]:
                raise ValueError("Label batch size does not match logit batch size.")

            loss_function = nn.CrossEntropyLoss()

            loss = loss_function(
                logits,
                labels.long(),
            )

            output["loss"] = loss

        return output

    # PROBABILITY HELPER
    @staticmethod
    def probabilities_from_logits(logits: torch.Tensor,) -> torch.Tensor:
        
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError("Expected logits with shape [batch_size, 2].")

        return torch.softmax(
            logits,
            dim=-1,
        )
