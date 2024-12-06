import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from huggingface_hub import login

from faesm.esmc import ESMC as FAESMC


# Function for benchmarking two models
def benchmark_flash_vs_official(sequence, use_flash_attn):
    # Flash Attention Implementation
    model_flash = FAESMC.from_pretrained("esmc_300m", use_flash_attn=use_flash_attn).to("cuda")
    input_ids_flash = model_flash.tokenizer(sequence, return_tensors="pt")["input_ids"].to("cuda")
    output_flash = model_flash(input_ids_flash)
    logits_flash = output_flash.sequence_logits
    embeddings_flash = output_flash.embeddings

    # Official Implementation
    protein = ESMProtein(sequence=sequence[0])  # Single sequence for now
    model_official = ESMC.from_pretrained("esmc_300m").to("cuda")
    protein_tensor = model_official.encode(protein)
    logits_output_official = model_official.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    logits_official = logits_output_official.logits.sequence
    embeddings_official = logits_output_official.embeddings

    # Compute differences
    logits_diff = torch.abs(logits_flash - logits_official).max()
    embeddings_diff = torch.abs(embeddings_flash - embeddings_official).max()

    return logits_diff.item(), embeddings_diff.item()


# Define the sequence
seq = "MPGWFKKAWYGLASLLSFSSFILIIVALVVPHWLSGKILCQTGVDLVNATDRELVKFIGDIYYGLFRGCKVRQCGLGGRQSQFTIFPHLVKELNAGLHVMILLLLFLALALALVSMGFAILNMIQVPYRAVSGPGGICLWNVLAGGVVALAIASFVAAVKFHDLTERIANFQEKLFQFVVVEEQYEESFWICVASASAHAANLVVVAISQIPLPEIKTKIEEATVTAEDILY"
sequence = [seq]

# Login to Hugging Face Hub (use your API key with "Read" permission)
login("YOUR_API_KEY")

# Benchmark with `use_flash_attn=True`
logits_diff_flash, embeddings_diff_flash = benchmark_flash_vs_official(
    sequence, use_flash_attn=True
)
print("[Flash Attention Enabled]")
print("Max absolute error in logits:", logits_diff_flash)
print("Max absolute error in embeddings:", embeddings_diff_flash)
assert logits_diff_flash < 1, f"Logits diff: {logits_diff_flash}"
assert embeddings_diff_flash < 0.1, f"Embeddings diff: {embeddings_diff_flash}"

# Benchmark with `use_flash_attn=False`
logits_diff_no_flash, embeddings_diff_no_flash = benchmark_flash_vs_official(
    sequence, use_flash_attn=False
)
print("\n[Flash Attention Disabled]")
print("Max absolute error in logits:", logits_diff_no_flash)
print("Max absolute error in embeddings:", embeddings_diff_no_flash)
assert logits_diff_no_flash < 1, f"Logits diff: {logits_diff_no_flash}"
assert embeddings_diff_no_flash < 0.1, f"Embeddings diff: {embeddings_diff_no_flash}"
