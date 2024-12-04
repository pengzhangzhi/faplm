
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
import torch
from transformers import EsmForMaskedLM, EsmTokenizer, EsmForProteinFolding
from tqdm import tqdm
from faesm.esm import FAEsmForMaskedLM
import random


def generate_random_protein_sequences(mini_length, max_length):
    """Generate random protein sequences."""
    length = random.randint(mini_length, max_length)
    return "".join(
        [
            random.choice("ACDEFGHIKLMNPQRSTVWY")
            for _ in range(length)
        ]
    )

# Set Seaborn theme and professional settings
sns.set_theme(style="white")  # Remove grid by using "white"
color_palette = sns.color_palette("Set2")  # Professional color palette

# Matplotlib font and size settings
plt.rcParams.update(
    {
        "font.family": "serif",  # Use serif fonts for a professional look
        "font.size": 14,  # Larger font size for better readability
        "axes.titlesize": 18,  # Larger titles
        "axes.labelsize": 16,  # Larger axis labels
        "xtick.labelsize": 14,  # Larger x-tick labels
        "ytick.labelsize": 14,  # Larger y-tick labels
        "legend.fontsize": 14,  # Larger legend font
    }
)


def benchmark_torch_memory(f, *args, **kwargs):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    f(*args, **kwargs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024  # Convert to GB


def benchmark_inference_time(f, *args, **kwargs):
    torch.cuda.synchronize()
    start_time = time.time()
    f(*args, **kwargs)
    torch.cuda.synchronize()
    return time.time() - start_time


def get_faesmfold(device):

    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", device_map="cpu").eval()
    model.esm = None
    model.esm = FAEsmForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D", use_fa=True).to(torch.float16).to(device).eval()
    model = model.to(device)
    return model


@pytest.mark.parametrize(
    "dtype,max_seq_lengths,repeats",
    [
        (
            torch.float16,
            [100,200,300,400,500],
            8,
        )
    ],
)
def test_esmfold_vs_faesmfold_benchmark(dtype, max_seq_lengths, repeats):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    esmfold = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()
    esmfold.esm = esmfold.esm.to(dtype)
    fa_esmfold = get_faesmfold(device)
    esm_memory_usage, fa_esm_memory_usage = [], []
    esm_inference_times, fa_esm_inference_times = [], []

    for seq_length in max_seq_lengths:
        inputs = generate_random_protein_sequences(mini_length=seq_length-50, max_length=seq_length)

        esm_memory_fold, fa_esm_memory_fold = [], []
        esm_time_fold, fa_esm_time_fold = [], []

        for _ in tqdm(range(repeats)):

            def esm_forward():
                
                esmfold.infer_pdb(inputs)
            esmfold.to(device)
            esm_memory_fold.append(benchmark_torch_memory(esm_forward))
            esm_time_fold.append(benchmark_inference_time(esm_forward))
            esmfold.to("cpu")
            torch.cuda.empty_cache()

            def fa_esm_forward():
                fa_esmfold.esm.half()
                fa_esmfold.infer_pdb(inputs)
            fa_esmfold.to(device)
            fa_esm_memory_fold.append(benchmark_torch_memory(fa_esm_forward))
            fa_esm_time_fold.append(benchmark_inference_time(fa_esm_forward))
            fa_esmfold.to("cpu")
            torch.cuda.empty_cache()

        esm_memory_usage.append(np.mean(esm_memory_fold))
        fa_esm_memory_usage.append(np.mean(fa_esm_memory_fold))
        esm_inference_times.append(np.mean(esm_time_fold))
        fa_esm_inference_times.append(np.mean(fa_esm_time_fold))

        print(
            f"Seq Len: {seq_length}, Avg ESMFold Mem: {esm_memory_usage[-1]:.3f} GB, Avg FAESMFold Mem: {fa_esm_memory_usage[-1]:.3f} GB"
        )
        print(
            f"Seq Len: {seq_length}, Avg ESMFold Time: {esm_inference_times[-1]:.3f} s, Avg FAESMFold Time: {fa_esm_inference_times[-1]:.3f} s"
        )

    max_seq_lengths_filtered = max_seq_lengths[1:]
    esm_inference_times = esm_inference_times[1:]
    fa_esm_inference_times = fa_esm_inference_times[1:]

    memory_reduction = [
        (1 - (fa / esm)) * 100 for fa, esm in zip(fa_esm_memory_usage, esm_memory_usage)
    ]
    time_reduction = [
        (1 - (fa / esm)) * 100 for fa, esm in zip(fa_esm_inference_times, esm_inference_times)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # Larger figure for better resolution

    # Left Plot: Memory Benchmark
    ax1 = axes[0]
    ax1.plot(
        max_seq_lengths,
        esm_memory_usage,
        label="ESMFold Memory Usage (GB)",
        marker="o",
        color=color_palette[0],
    )
    ax1.plot(
        max_seq_lengths,
        fa_esm_memory_usage,
        label="FAESMFold Memory Usage (GB)",
        marker="o",
        color=color_palette[1],
    )
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Memory Usage (GB)", color=color_palette[0])
    ax1.tick_params(axis="y", labelcolor=color_palette[0])
    ax1.legend(loc="upper left")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        max_seq_lengths,
        memory_reduction,
        label="Memory Reduction (%)",
        marker="o",
        linestyle="--",
        color=color_palette[2],
    )
    ax1_twin.set_ylabel("Memory Reduction (%)", color=color_palette[2])
    ax1_twin.tick_params(axis="y", labelcolor=color_palette[2])
    ax1_twin.legend(loc="upper right")

    ax1.set_title("Memory Benchmark")

    # Right Plot: Time Benchmark
    ax2 = axes[1]
    ax2.plot(
        max_seq_lengths_filtered,
        esm_inference_times,
        label="ESMFold Inference Time (s)",
        marker="o",
        color=color_palette[0],
    )
    ax2.plot(
        max_seq_lengths_filtered,
        fa_esm_inference_times,
        label="FAESMFold Inference Time (s)",
        marker="o",
        color=color_palette[1],
    )
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Inference Time (s)", color=color_palette[0])
    ax2.tick_params(axis="y", labelcolor=color_palette[0])
    ax2.legend(loc="upper left")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        max_seq_lengths_filtered,
        time_reduction,
        label="Time Reduction (%)",
        marker="o",
        linestyle="--",
        color=color_palette[2],
    )
    ax2_twin.set_ylabel("Time Reduction (%)", color=color_palette[2])
    ax2_twin.tick_params(axis="y", labelcolor=color_palette[2])
    ax2_twin.legend(loc="upper right")

    ax2.set_title("Inference Time Benchmark")

    plt.suptitle(
        f"Data Type: {dtype}, Averaged over {repeats} runs",
        fontsize=20,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("esmfold_benchmark.png", dpi=300)  # High resolution
    plt.close()

    for seq_length, fa_mem, esm_mem, fa_time, esm_time in zip(
        max_seq_lengths,
        fa_esm_memory_usage,
        esm_memory_usage,
        fa_esm_inference_times,
        esm_inference_times,
    ):
        assert (
            fa_mem <= esm_mem
        ), f"Seq {seq_length}: FAESM {fa_mem:.3f} GB > ESM {esm_mem:.3f} GB!"
        assert (
            fa_time <= esm_time
        ), f"Seq {seq_length}: FAESM {fa_time:.3f} s > ESM {esm_time:.3f} s!"
