
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from einops import rearrange
from faesm.esm import *


def generate_random_esm2_inputs(
    tokenizer,
    batch_size=3,
    min_seq_length=5,
    max_seq_length=10,
    device="cuda"
):
    """
    Generate random sequences of amino acids and tokenize them for input into an ESM2 model.

    Args:
        batch_size (int): Number of sequences in the batch.
        min_seq_length (int): Minimum sequence length.
        max_seq_length (int): Maximum sequence length.
        tokenizer (obj): The tokenizer object for ESM2.
        device (str): Device to store tensors ('cpu' or 'cuda').

    Returns:
        dict: Tokenized input sequences ready for ESM2 model.
    """

    # Generate random lengths for each sequence in the batch
    random_lengths = torch.randint(
        min_seq_length, 
        max_seq_length + 1, 
        (batch_size,), 
        device=device
    )

    # Generate random sequences
    random_tokens = [
        torch.randint(low=4, high=29, size=(length,), device=device).tolist()
        for length in random_lengths
    ]

    # Convert token IDs to strings
    sequences = ["".join(tokenizer.convert_ids_to_tokens(seq)) for seq in random_tokens]

    # Tokenize the sequences
    esm_input = tokenizer.batch_encode_plus(
        sequences,
        add_special_tokens=True,  # Adds [CLS] and [SEP] tokens
        padding=True,  # Pad to the maximum sequence length in the batch
        truncation=True,  # Truncate sequences longer than the maximum model input length
        return_tensors="pt"
    )
    
    # Move tensors to the specified device
    esm_input = {k: v.to(device) for k, v in esm_input.items()}

    return esm_input



def test_esm_vs_faesm_numberic(
    tokenizer_name="facebook/esm2_t33_650M_UR50D",
    batch_size=10,
    min_seq_length=3,
    max_seq_length=10,
    use_fa=True,
    dtype=torch.float16
):
    """
    Simple test function to compare ESM and FAESM outputs with configurable dtype.

    Args:
        tokenizer_name (str): The name of the tokenizer and pretrained model.
        batch_size (int): Number of sequences in the batch.
        min_seq_length (int): Minimum sequence length.
        max_seq_length (int): Maximum sequence length.
        use_fa (bool): Whether to use flash attention in FAESM.
        dtype (torch.dtype): Data type for model and inputs (e.g., torch.float16, torch.float32).
    """
    import torch
    from einops import rearrange
    from transformers import EsmTokenizer, EsmForMaskedLM

    print(f"Testing with dtype: {dtype}")

    # Load the tokenizer and models
    tokenizer = EsmTokenizer.from_pretrained(tokenizer_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    esm = EsmForMaskedLM.from_pretrained(tokenizer_name)
    esm = esm.to(device).to(dtype)
    esm.eval()

    fa_esm = FAEsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", use_fa=use_fa)
    fa_esm = fa_esm.to(device).to(dtype)
    fa_esm.eval()

    # Generate random inputs
    inputs = generate_random_esm2_inputs(
        tokenizer,
        batch_size=batch_size,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        device=device
    )

    # Identify padding token mask
    padding_mask = inputs["attention_mask"]

    # Get predictions
    esm_output = esm(**inputs, output_hidden_states=True)
    esm_logits = esm_output.logits
    esm_repr = esm_output.hidden_states[-1]  # Last hidden state as representation

    fa_esm_output = fa_esm(inputs["input_ids"])
    fa_esm_logits = fa_esm_output['logits']
    fa_esm_repr = fa_esm_output['last_hidden_state']

    # Ensure logits have consistent data type
    logit_mask = rearrange(~(padding_mask == 1), "b s -> b s 1").bool()
    esm_logits = esm_logits.to(dtype).masked_fill(logit_mask, 0.0)
    fa_esm_logits = fa_esm_logits.to(dtype).masked_fill(logit_mask, 0.0)

    # Compare logits
    print("### Comparisons between model logits ###\n")
    print("ESM vs FAESM:")
    print(f"All close: {torch.allclose(esm_logits, fa_esm_logits, atol=1e-5)}")
    print(f"Max absolute difference: {torch.abs(esm_logits - fa_esm_logits).max()}")
    print(f"Mean absolute difference: {torch.abs(esm_logits - fa_esm_logits).mean()}\n")

    # Compare representations
    print("### Comparisons between model representations ###\n")
    esm_repr, fa_esm_repr = map(lambda x: x.masked_fill(logit_mask, 0.0), (esm_repr, fa_esm_repr))
    repr_diff = esm_repr - fa_esm_repr

    print(f"All close: {torch.allclose(esm_repr, fa_esm_repr, atol=1e-5)}")
    print(f"Max absolute difference: {repr_diff.abs().max()}")
    print(f"Mean absolute difference: {repr_diff.abs().mean()}")
    print(f"Standard deviation of differences: {repr_diff.abs().std()}\n")

if __name__ == '__main__':
    test_esm_vs_faesm_numberic()