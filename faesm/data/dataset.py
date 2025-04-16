import random
import math
from collections import Counter
import numpy as np
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def calculate_entropy(seq):
    counts = Counter(seq)
    total = len(seq)
    probabilities = np.array(list(counts.values())) / total
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

class MaxTokenCollater(object):
    def __init__(self, tokenizer_path=None, max_tokens=6000, max_len=1022, min_entropy=None):
        """
        Collator for dynamically batching sequences.

        Parameters:
        - tokenizer_path: Path to the tokenizer or pretrained model (e.g., EsmTokenizer).
        - max_tokens: Maximum number of tokens for the batch.
        - max_len: Maximum length of each sequence.
        - min_entropy: Minimum entropy threshold for sequences.
        """
        self.max_tokens = max_tokens
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or "facebook/esm2_t30_150M_UR50D")
        self.min_entropy = min_entropy

    def __call__(self, batch):
        """
        Processes a list of sequences:
        1) Shuffles the list.
        2) Truncates sequences that exceed self.max_len.
        3) Computes entropy for the truncated sequences.
        4) (Optional) Filters out sequences with entropy < self.min_entropy.
        5) Sorts sequences by descending entropy.
        6) Includes as many sequences as possible up to self.max_tokens.
        7) Tokenizes and returns the collated batch.
        
        Args:
        - batch: A list of dicts or strings containing the "sequence" to tokenize.
        
        Returns:
        - A dictionary with:
            "input_ids": Tensor of token IDs
            "attention_mask": Tensor of booleans
            "targets": (Optional) clone of "input_ids"
        """
        # 1) Validate and extract sequences
        if isinstance(batch[0], dict):
            sequences = [item["sequence"] for item in batch]
        elif isinstance(batch[0], str):
            sequences = batch
        else:
            raise ValueError(f"Invalid batch format {type(batch[0])}!")
        
        if not sequences:
            raise ValueError("Empty sequence list provided to collator!")

        # 2) Shuffle
        random.shuffle(sequences)

        # 3) Truncate each sequence if longer than max_len, then compute entropy
        truncated_seqs_with_entropy = []
        for seq in sequences:
            # Truncate if necessary
            if len(seq) > self.max_len:
                start_idx = random.randint(0, len(seq) - self.max_len)
                seq = seq[start_idx : start_idx + self.max_len]
            
            # Compute entropy on the truncated sequence
            seq_entropy = calculate_entropy(seq)

            # If min_entropy is set, skip sequences that don't meet the threshold
            if self.min_entropy is not None and seq_entropy < self.min_entropy:
                print(f'[debug] seq: {seq}, seq_entropy: {seq_entropy}')

            truncated_seqs_with_entropy.append((seq, seq_entropy))

        # 4) Sort by descending entropy
        truncated_seqs_with_entropy.sort(key=lambda x: x[1], reverse=True)

        # 5) Perform cumulative sum of lengths and cutoff
        cumulative = 0
        final_sequences = []
        seq_lengths = []

        for seq, ent in truncated_seqs_with_entropy:
            seq_len = len(seq)
            if cumulative + seq_len <= self.max_tokens:
                # If it fully fits, add it
                final_sequences.append(seq)
                seq_lengths.append(seq_len)
                cumulative += seq_len
            else:
                # If partial fit is possible
                remaining_tokens = self.max_tokens - cumulative
                if remaining_tokens > 0:
                    # Truncate the last sequence to fit remaining tokens
                    final_sequences.append(seq[:remaining_tokens])
                    seq_lengths.append(remaining_tokens)
                    cumulative += remaining_tokens
                # Then break, as no more tokens left
                break

        # 6) Tokenize the final truncated list of sequences
        tokenized_batch = self.tokenizer(
            final_sequences,
            add_special_tokens=True,
            padding="longest",
            truncation=False,  # No further truncation; manually handled above
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized_batch["input_ids"],
            "attention_mask": tokenized_batch["attention_mask"].bool(),
            "targets": tokenized_batch["input_ids"].clone(),
        }
# ---------------------------
# Define the setup_dataloader Function
# ---------------------------

def setup_dataloader(
        mixed_dataset,  
        max_tokens=6000,
        max_len=1022,
        num_workers=4,
        pin_memory=True,
        min_entropy=None,
        **kwargs,
    ) -> DataLoader:
    """
    Sets up a DataLoader for the mixed dataset using the provided collator.

    Parameters:
    - mixed_dataset: The combined HuggingFace Dataset.
    - collater: An instance of the MaxTokenCollater class.
    - batch_size: Number of samples per batch.
    - num_workers: Number of subprocesses to use for data loading.
    - shuffle: Whether to shuffle the dataset each epoch.
    - pin_memory: Whether to pin memory in DataLoader.
    - min_entropy: Minimum entropy threshold for sequences.
    Returns:
    - A PyTorch DataLoader.
    """
    collater = MaxTokenCollater(max_tokens=max_tokens, max_len=max_len, min_entropy=min_entropy)
    dataloader = DataLoader(
        dataset=mixed_dataset,
        batch_size=min(max_tokens//20, 200),
        num_workers=num_workers,
        collate_fn=collater,
        pin_memory=pin_memory,
        **kwargs,
    )
    return dataloader


def load_mix_dataset(
    streaming=True,
    mix_probabilities=[0.8, 0.2],
    seed=None,
):
    """
    Loads and mixes datasets with specified probabilities, ensuring the 'sequence' column
    is standardized to 'large_string' for compatibility.

    Parameters:
    - split: The split of the datasets to load (e.g., "train", "validation", "test").
    - streaming: Whether to load the datasets in streaming mode.
    - mix_probabilities: List of probabilities for mixing datasets (e.g., [0.8, 0.2]).
    - seed: Seed for reproducibility during dataset mixing.

    Returns:
    - mixed_dataset: A mixed HuggingFace dataset with aligned keys and types.
    """
    from datasets import Features, Value
    split="train"
    # Step 1: Load the datasets
    print(f"Loading UniRef50 dataset ({split}, streaming={streaming})...")
    uniref50 = load_dataset("zhangzhi/Uniref50", split=split, streaming=streaming)

    print(f"Loading OMG_prot50 dataset ({split}, streaming={streaming})...")
    omg_prot50 = load_dataset("tattabio/OMG_prot50", split=split, streaming=streaming)
    # columns are "sequence"  and "length"
    uniref50 = uniref50.cast(Features({"sequence": Value("large_string"), "length": Value("int32")}))
    # omg_prot50 = omg_prot50.cast(Features({"sequence": Value("large_string")}))

    # Step 3: Mix datasets with the desired probabilities
    print("Mixing datasets with probabilities:", mix_probabilities)
    mixed_dataset = interleave_datasets(
        [uniref50, omg_prot50],
        probabilities=mix_probabilities,
        seed=seed  # Use the seed for reproducibility
    )
    # shuffle mixed_dataset
    mixed_dataset = mixed_dataset.shuffle(
        seed=seed,
        buffer_size=40_000_000,
    )
    print("Shuffling mixed dataset...")
    

    return mixed_dataset


def main():
    # Configuration
    max_tokens = 6000  # Example max token limit per batch
    batch_size = 32
    num_workers = 4
    shuffle = True
    mix_probabilities = [0.2, 0.8]  # 20% UniRef50, 80% OMG_prot50
    # Example usage
    mixed_dataset = load_mix_dataset(
        streaming=True,
        mix_probabilities=[0.7, 0.3],  # 70% UniRef50, 30% OMG_prot50
        seed=123  # Custom seed for reproducibility
    )

    # Preview the mixed dataset
    import torch
    pct_omg = torch.tensor(['id' in item for item in mixed_dataset.take(2000)]).sum()/2000
    print(f'OMG_prot50 percentage: {pct_omg:.2%}')
    # Step 4: Setup DataLoader
    print("Setting up DataLoader...")
    dataloader = setup_dataloader(
        mixed_dataset=mixed_dataset,
        num_workers=num_workers,
    )

    # Step 5: (Optional) Iterate through a few batches to test
    print("Iterating through the first few batches...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print("Input IDs:", batch["input_ids"].shape)
        print("Attention Mask:", batch["attention_mask"].shape)
        print("Targets:", batch["targets"].shape)
        if batch_idx >= 2:  # Display first 3 batches
            break

    print("DataLoader setup complete. Ready for training or further processing.")

if __name__ == "__main__":
    main()
