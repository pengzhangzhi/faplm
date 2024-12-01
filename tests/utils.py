import os
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from faesm.esm import *





def generate_random_esm2_inputs(
    tokenizer,
    batch_size=3,
    min_seq_length=5,
    max_seq_length=10,
    device="cuda"
):
    """Generate random ESM2 model inputs."""
    random_lengths = torch.randint(
        min_seq_length, 
        max_seq_length + 1, 
        (batch_size,), 
        device=device
    )
    random_tokens = [
        torch.randint(low=4, high=29, size=(length,), device=device).tolist()
        for length in random_lengths
    ]
    sequences = ["".join(tokenizer.convert_ids_to_tokens(seq)) for seq in random_tokens]
    esm_input = tokenizer.batch_encode_plus(
        sequences,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    esm_input = {k: v.to(device) for k, v in esm_input.items()}
    return esm_input

