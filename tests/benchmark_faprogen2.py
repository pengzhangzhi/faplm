import time

import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
import torch
from transformers import AutoTokenizer

from faesm.progen2 import ProGenForCausalLM as FAProGenForCausalLM

try:
    from tests.progen2.models.progen.modeling_progen import ProGenForCausalLM
except ImportError:
    readme = """
To install the original ProGen2, clone the repository and move the `progen2` directory to this tests directory:

```bash
git clone https://github.com/salesforce/progen
mv progen/progen2 ./
rm -rf progen
```

with the progen2 directory in the tests directory, you can run the tests involving ProGen2 in `benchmark_faprogen2.py`.
    """
    raise ImportError(
        "Please download the ProGen2 model following the instructions in the README.md file:\n{}".format(
            readme
        )
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


def generate_random_protein_sequences(mini_length, max_length):
    import random

    """Generate random protein sequences."""
    length = random.randint(mini_length, max_length)
    return "".join([random.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(length)])


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
    "model_version,dtype,max_seq_lengths,repeats",
    [
        (
            "jinyuan22/ProGen2-base",
            torch.float16,
            [600, 900, 1200, 1500, 1800, 2000],
            5,
        )
    ],
)
def test_progen2_vs_faprogen2_benchmark(model_version, dtype, max_seq_lengths, repeats):
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    device1 = "cuda" if torch.cuda.is_available() else "cpu"
    # device2 = "cuda:1" if torch.cuda.is_available() else "cpu"

    progen2 = ProGenForCausalLM.from_pretrained(model_version).to(dtype).to("cpu").eval()
    fa_progen2 = FAProGenForCausalLM.from_pretrained(model_version).to(dtype).to("cpu").eval()

    progen2_memory_usage, fa_progen2_memory_usage = [], []
    progen2_inference_times, fa_progen2_inference_times = [], []

    for seq_length in max_seq_lengths:
        sequence = generate_random_protein_sequences(
            mini_length=seq_length - 50, max_length=seq_length
        )
        inputs1 = tokenizer(sequence, return_tensors="pt").to(device1)
        # inputs2 = tokenizer(sequence, return_tensors="pt").to(device1)

        progen2_memory_fold, fa_progen2_memory_fold = [], []
        progen2_time_fold, fa_progen2_time_fold = [], []

        for _ in range(repeats):

            def progen2_forward():
                progen2(inputs1.input_ids)

            progen2.to(device1)
            progen2_memory_fold.append(benchmark_torch_memory(progen2_forward))
            progen2_time_fold.append(benchmark_inference_time(progen2_forward))
            progen2.to("cpu")

            def fa_progen2_forward():
                fa_progen2(inputs1.input_ids)

            fa_progen2.to(device1)
            fa_progen2_memory_fold.append(benchmark_torch_memory(fa_progen2_forward))
            fa_progen2_time_fold.append(benchmark_inference_time(fa_progen2_forward))
            fa_progen2.to("cpu")

        progen2_memory_usage.append(np.mean(progen2_memory_fold))
        fa_progen2_memory_usage.append(np.mean(fa_progen2_memory_fold))
        progen2_inference_times.append(np.mean(progen2_time_fold))
        fa_progen2_inference_times.append(np.mean(fa_progen2_time_fold))

        print(
            f"Seq Len: {seq_length}, Avg progen2 Mem: {progen2_memory_usage[-1]:.3f} GB, Avg FAprogen2 Mem: {fa_progen2_memory_usage[-1]:.3f} GB"
        )
        print(
            f"Seq Len: {seq_length}, Avg progen2 Time: {progen2_inference_times[-1]:.3f} s, Avg FAprogen2 Time: {fa_progen2_inference_times[-1]:.3f} s"
        )

    max_seq_lengths_filtered = max_seq_lengths[1:]
    progen2_inference_times = progen2_inference_times[1:]
    fa_progen2_inference_times = fa_progen2_inference_times[1:]

    memory_reduction = [
        (1 - (fa / progen2)) * 100
        for fa, progen2 in zip(fa_progen2_memory_usage, progen2_memory_usage)
    ]
    time_reduction = [
        (1 - (fa / progen2)) * 100
        for fa, progen2 in zip(fa_progen2_inference_times, progen2_inference_times)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # Larger figure for better resolution

    # Left Plot: Memory Benchmark
    ax1 = axes[0]
    ax1.plot(
        max_seq_lengths,
        progen2_memory_usage,
        label="progen2 Memory Usage (GB)",
        marker="o",
        color=color_palette[0],
    )
    ax1.plot(
        max_seq_lengths,
        fa_progen2_memory_usage,
        label="FAprogen2 Memory Usage (GB)",
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
        progen2_inference_times,
        label="progen2 Inference Time (s)",
        marker="o",
        color=color_palette[0],
    )
    ax2.plot(
        max_seq_lengths_filtered,
        fa_progen2_inference_times,
        label="FAprogen2 Inference Time (s)",
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
        f"Model Version: {model_version}\nBatch Size: 1, Data Type: {dtype}, Averaged over {repeats} runs",
        fontsize=20,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("FAProGen2_benchmark.png", dpi=300)  # High resolution
    plt.close()

    for seq_length, fa_mem, progen2_mem, fa_time, progen2_time in zip(
        max_seq_lengths,
        fa_progen2_memory_usage,
        progen2_memory_usage,
        fa_progen2_inference_times,
        progen2_inference_times,
    ):
        assert (
            fa_mem <= progen2_mem
        ), f"Seq {seq_length}: FAprogen2 {fa_mem:.3f} GB > progen2 {progen2_mem:.3f} GB!"
        assert (
            fa_time <= progen2_time
        ), f"Seq {seq_length}: FAprogen2 {fa_time:.3f} s > progen2 {progen2_time:.3f} s!"
