import torch
import torch.utils.benchmark as benchmark
from einops import rearrange
from transformers import EsmTokenizer, EsmForMaskedLM
from tests.test_compare_esm import *


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    """
    Helper function to benchmark a PyTorch function and return execution time in microseconds.
    """
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6


def test_esm_vs_faesm_benchmark(
    tokenizer_name="facebook/esm2_t6_8M_UR50D",
    batch_size=128,
    min_seq_length=200,
    max_seq_length=700,
    dtype=torch.float16,
):
    """
    Benchmark the execution time of ESM and FAESM models.

    Args:
        tokenizer_name (str): The name of the tokenizer and pretrained model.
        batch_size (int): Number of sequences in the batch.
        min_seq_length (int): Minimum sequence length.
        max_seq_length (int): Maximum sequence length.
        use_fa (bool): Whether to use flash attention in FAESM.
        dtype (torch.dtype): Data type for model and inputs (e.g., torch.float16, torch.float32).
        num_runs (int): Number of runs for benchmarking.
    """
    print(f"Benchmarking with dtype: {dtype}")

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

    esm_time = benchmark_torch_function_in_microseconds(esm_forward)

    # Benchmark FAESM
    def fa_esm_forward():
        fa_esm(inputs["input_ids"])

    fa_esm_time = benchmark_torch_function_in_microseconds(fa_esm_forward)

    # Report benchmark results
    print("\n### Benchmark Results ###")
    print(f"ESM average time: {esm_time:.3f} microseconds")
    print(f"FAESM average time: {fa_esm_time:.3f} microseconds")
    print(f"FAESM is {(fa_esm_time / esm_time) * 100:.2f}% of ESM time\n")

if __name__ == '__main__':
    test_esm_vs_faesm_benchmark()