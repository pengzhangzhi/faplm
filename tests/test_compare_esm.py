import os

import matplotlib.pyplot as plt
import pytest
import torch
from einops import rearrange
from transformers import EsmForMaskedLM, EsmTokenizer

from faesm.esm import FAEsmForMaskedLM
from tests.utils import generate_random_esm2_inputs


@pytest.fixture(scope="module")
def setup_output_dir():
    """Fixture to set up the output directory for saving figures."""
    output_dir = "tests/fig/error"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_differences(data, title, ylabel, tokenizing_names, save_path=None):
    """Plot differences for logits and representations.

    Args:
        data (list): List of lists containing the differences for each tokenizing name.
        title (str): Title of the plot.
        ylabel (str): Label for the y-axis.
        tokenizing_names (list): Tokenizing names for the x-axis.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, diffs in data.items():
        ax.bar(tokenizing_names, diffs, label=model)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("ESM Checkpoint")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


@pytest.mark.parametrize(
    "tokenizing_names",
    [
        [
            "facebook/esm2_t36_3B_UR50D",
            "facebook/esm2_t33_650M_UR50D",
            "facebook/esm2_t30_150M_UR50D",
        ],
    ],
)
def test_esm_vs_faesm_numeric(
    tokenizing_names,
    setup_output_dir,
    batch_size=10,
    min_seq_length=10,
    max_seq_length=200,
    use_fa=True,
    dtype=torch.float16,
):
    """Test function to compare numeric differences between ESM and FAESM outputs, and assert that
    the max and mean differences are within expected ranges."""
    results = {
        "max_diff_logits": {},
        "mean_diff_logits": {},
        "max_diff_repr": {},
        "mean_diff_repr": {},
    }

    for tokenizer_name in tokenizing_names:
        tokenizer = EsmTokenizer.from_pretrained(tokenizer_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        esm = EsmForMaskedLM.from_pretrained(tokenizer_name)
        esm = esm.to(device).to(dtype)
        esm.eval()

        fa_esm = FAEsmForMaskedLM.from_pretrained(tokenizer_name, use_fa=use_fa)
        fa_esm = fa_esm.to(device).to(dtype)
        fa_esm.eval()

        inputs = generate_random_esm2_inputs(
            tokenizer,
            batch_size=batch_size,
            min_seq_length=min_seq_length,
            max_seq_length=max_seq_length,
            device=device,
        )
        padding_mask = inputs["attention_mask"]

        esm_output = esm(**inputs, output_hidden_states=True)
        esm_logits = esm_output.logits
        esm_repr = esm_output.hidden_states[-1]

        fa_esm_output = fa_esm(inputs["input_ids"])
        fa_esm_logits = fa_esm_output["logits"]
        fa_esm_repr = fa_esm_output["last_hidden_state"]

        logit_mask = rearrange(~(padding_mask == 1), "b s -> b s 1").bool()
        esm_logits = esm_logits.to(dtype).masked_fill(logit_mask, 0.0)
        fa_esm_logits = fa_esm_logits.to(dtype).masked_fill(logit_mask, 0.0)

        esm_repr, fa_esm_repr = map(
            lambda x: x.masked_fill(logit_mask, 0.0), (esm_repr, fa_esm_repr)
        )

        logit_diff = torch.abs(esm_logits - fa_esm_logits)
        repr_diff = torch.abs(esm_repr - fa_esm_repr)

        results["max_diff_logits"][tokenizer_name] = logit_diff.max().item()
        results["mean_diff_logits"][tokenizer_name] = logit_diff.mean().item()
        results["max_diff_repr"][tokenizer_name] = repr_diff.max().item()
        results["mean_diff_repr"][tokenizer_name] = repr_diff.mean().item()

    # Assertions for maximum and mean differences
    max_repr_diff = max(results["max_diff_repr"].values())
    mean_repr_diff = max(results["mean_diff_repr"].values())
    max_logits_diff = max(results["max_diff_logits"].values())
    mean_logits_diff = max(results["mean_diff_logits"].values())
    assert max_logits_diff < 0.8, f"Max logits difference too large: {max_logits_diff}"
    assert mean_logits_diff < 0.05, f"Mean logits difference too large: {mean_logits_diff}"
    assert max_repr_diff < 0.7, f"Max representation difference too large: {max_repr_diff}"
    assert mean_repr_diff < 0.05, f"Mean representation difference too large: {mean_repr_diff}"
