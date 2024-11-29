import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from tests.test_compare_esm import *


def benchmark_torch_memory(f, *args, **kwargs):
    """
    Helper function to benchmark GPU memory usage of a PyTorch function.
    Returns:
        max_memory (float): Maximum memory allocated during the function call in MB.
    """
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Run the function
    f(*args, **kwargs)
    torch.cuda.synchronize()

    # Get peak memory usage
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
    return peak_memory


def test_esm_vs_faesm_memory_benchmark(
    tokenizer_name="facebook/esm2_t6_8M_UR50D",
    batch_size=128,
    min_seq_length=200,
    max_seq_length=700,
    dtype=torch.float16,
):
    """
    Benchmark the memory usage of ESM and FAESM models during forward pass.

    Args:
        tokenizer_name (str): The name of the tokenizer and pretrained model.
        batch_size (int): Number of sequences in the batch.
        min_seq_length (int): Minimum sequence length.
        max_seq_length (int): Maximum sequence length.
        dtype (torch.dtype): Data type for model and inputs (e.g., torch.float16, torch.float32).
    """
    print(f"Benchmarking memory with dtype: {dtype}")

    # Load the tokenizer and models
    tokenizer = EsmTokenizer.from_pretrained(tokenizer_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    esm = EsmForMaskedLM.from_pretrained(tokenizer_name)
    esm = esm.to(device).to(dtype)
    esm.eval()

    fa_esm = FAEsmForMaskedLM.from_pretrained(tokenizer_name, use_fa=True)
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

    # Benchmark ESM
    def esm_forward():
        esm(**inputs, output_hidden_states=False)

    esm_memory = benchmark_torch_memory(esm_forward)

    # Benchmark FAESM
    def fa_esm_forward():
        fa_esm(inputs["input_ids"])

    fa_esm_memory = benchmark_torch_memory(fa_esm_forward)

    # Report benchmark results
    print("\n### Memory Benchmark Results ###")
    print(f"ESM peak memory usage: {esm_memory:.3f} MB")
    print(f"FAESM peak memory usage: {fa_esm_memory:.3f} MB")
    print(f"FAESM uses {(fa_esm_memory / esm_memory) * 100:.2f}% of ESM memory\n")


if __name__ == '__main__':
    test_esm_vs_faesm_memory_benchmark()