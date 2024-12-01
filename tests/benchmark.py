import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import matplotlib.pyplot as plt
import seaborn as sns
import pytest
from tests.utils import generate_random_esm2_inputs
from faesm.esm import FAEsmForMaskedLM
import time
import numpy as np

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


@pytest.mark.parametrize(
    "model_version,batch_size,dtype,max_seq_lengths,repeats",
    [
        (
            "facebook/esm2_t33_650M_UR50D",
            8,
            torch.float16,
            [100, 200, 300, 400, 500, 600, 700, 800, 1000],
            10,
        )
    ],
)
def test_esm_vs_faesm_benchmark(
    model_version, batch_size, dtype, max_seq_lengths, repeats
):
    tokenizer = EsmTokenizer.from_pretrained(model_version)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    esm = EsmForMaskedLM.from_pretrained(model_version).to(device).to(dtype).eval()
    fa_esm = (
        FAEsmForMaskedLM.from_pretrained(model_version, use_fa=True)
        .to(device)
        .to(dtype)
        .eval()
    )

    esm_memory_usage, fa_esm_memory_usage = [], []
    esm_inference_times, fa_esm_inference_times = [], []

    for seq_length in max_seq_lengths:
        inputs = generate_random_esm2_inputs(
            tokenizer,
            batch_size=batch_size,
            min_seq_length=50,
            max_seq_length=seq_length,
            device=device,
        )

        esm_memory_fold, fa_esm_memory_fold = [], []
        esm_time_fold, fa_esm_time_fold = [], []

        for _ in range(repeats):

            def esm_forward():
                esm(**inputs, output_hidden_states=False)

            esm_memory_fold.append(benchmark_torch_memory(esm_forward))
            esm_time_fold.append(benchmark_inference_time(esm_forward))

            def fa_esm_forward():
                fa_esm(inputs["input_ids"])

            fa_esm_memory_fold.append(benchmark_torch_memory(fa_esm_forward))
            fa_esm_time_fold.append(benchmark_inference_time(fa_esm_forward))

        esm_memory_usage.append(np.mean(esm_memory_fold))
        fa_esm_memory_usage.append(np.mean(fa_esm_memory_fold))
        esm_inference_times.append(np.mean(esm_time_fold))
        fa_esm_inference_times.append(np.mean(fa_esm_time_fold))

        print(
            f"Seq Len: {seq_length}, Avg ESM Mem: {esm_memory_usage[-1]:.3f} GB, Avg FAESM Mem: {fa_esm_memory_usage[-1]:.3f} GB"
        )
        print(
            f"Seq Len: {seq_length}, Avg ESM Time: {esm_inference_times[-1]:.3f} s, Avg FAESM Time: {fa_esm_inference_times[-1]:.3f} s"
        )

    max_seq_lengths_filtered = max_seq_lengths[1:]
    esm_inference_times = esm_inference_times[1:]
    fa_esm_inference_times = fa_esm_inference_times[1:]

    memory_reduction = [
        (1 - (fa / esm)) * 100 for fa, esm in zip(fa_esm_memory_usage, esm_memory_usage)
    ]
    time_reduction = [
        (1 - (fa / esm)) * 100
        for fa, esm in zip(fa_esm_inference_times, esm_inference_times)
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(20, 8)
    )  # Larger figure for better resolution

    # Left Plot: Memory Benchmark
    ax1 = axes[0]
    ax1.plot(
        max_seq_lengths,
        esm_memory_usage,
        label="ESM Memory Usage (GB)",
        marker="o",
        color=color_palette[0],
    )
    ax1.plot(
        max_seq_lengths,
        fa_esm_memory_usage,
        label="FAESM Memory Usage (GB)",
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
        label="ESM Inference Time (s)",
        marker="o",
        color=color_palette[0],
    )
    ax2.plot(
        max_seq_lengths_filtered,
        fa_esm_inference_times,
        label="FAESM Inference Time (s)",
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
        f"Model Version: {model_version}\nBatch Size: {batch_size}, Data Type: {dtype}, Averaged over {repeats} runs",
        fontsize=20,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("benchmark.png", dpi=300)  # High resolution
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
