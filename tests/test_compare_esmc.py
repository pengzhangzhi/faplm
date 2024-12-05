import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from faesm.esmc import ESMC as FAESMC

# Define the sequence
seq = "MPGWFKKAWYGLASLLSFSSFILIIVALVVPHWLSGKILCQTGVDLVNATDRELVKFIGDIYYGLFRGCKVRQCGLGGRQSQFTIFPHLVKELNAGLHVMILLLLFLALALALVSMGFAILNMIQVPYRAVSGPGGICLWNVLAGGVVALAIASFVAAVKFHDLTERIANFQEKLFQFVVVEEQYEESFWICVASASAHAANLVVVAISQIPLPEIKTKIEEATVTAEDILY"
sequence = [seq]

# Flash Attention Implementation
model_flash = FAESMC.from_pretrained("esmc_300m", use_flash_attn=False).to("cuda")
input_ids_flash = model_flash.tokenizer(sequence, return_tensors="pt")["input_ids"].to("cuda")
output_flash = model_flash(input_ids_flash)
logits_flash = output_flash.sequence_logits
embeddings_flash = output_flash.embeddings

# Official Implementation
from huggingface_hub import login

# Will instruct you how to get an API key from huggingface hub, make one with "Read" permission.
login("Your API key")
protein = ESMProtein(sequence=seq)
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

# Print results
print("Max absolute error in logits:", logits_diff.item())
print("Max absolute error in embeddings:", embeddings_diff.item())
assert logits_diff < 1, f"Logits diff: {logits_diff}"
assert embeddings_diff < 0.1, f"Embeddings diff: {embeddings_diff}"
