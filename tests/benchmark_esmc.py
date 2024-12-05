import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pytest
import torch
from esm.models.esmc import ESMC
from faesm.esmc import ESMC as FAESMC

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


# Helper Functions
def generate_random_esm2_inputs(tokenizer, batch_size=3, min_seq_length=5, max_seq_length=10, device="cuda"):
    random_lengths = torch.randint(min_seq_length, max_seq_length + 1, (batch_size,), device=device)
    random_tokens = [
        torch.randint(low=4, high=29, size=(length,), device=device).tolist() for length in random_lengths
    ]
    sequences = ["".join(tokenizer.convert_ids_to_tokens(seq)) for seq in random_tokens]
    esm_input = tokenizer.batch_encode_plus(
        sequences,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    esm_input = {k: v.to(device) for k, v in esm_input.items()}
    return esm_input


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


# PyTest Fixture
@pytest.fixture
def setup_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "esmc_300m"
    esm_model = ESMC.from_pretrained(model_name).to(device).eval()
    fa_esm_model = FAESMC.from_pretrained(model_name, use_flash_attn=True).to(device).eval()
    tokenizer = fa_esm_model.tokenizer
    return esm_model, fa_esm_model, tokenizer, device


# Test Function
def test_benchmark_esmc(setup_models):
    esm_model, fa_esm_model, tokenizer, device = setup_models

    # Benchmark parameters
    batch_size = 16
    dtype = torch.bfloat16
    max_seq_lengths = torch.arange(100, 2000, 50).tolist()
    repeats = 10

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
        input_ids = inputs["input_ids"]
        fa_inputs = input_ids
        esm_memory_fold, fa_esm_memory_fold = [], []
        esm_time_fold, fa_esm_time_fold = [], []

        for _ in range(repeats):

            def esm_forward():
                esm_model(input_ids)

            esm_memory_fold.append(benchmark_torch_memory(esm_forward))
            esm_time_fold.append(benchmark_inference_time(esm_forward))

            def fa_esm_forward():
                fa_esm_model(fa_inputs)

            fa_esm_memory_fold.append(benchmark_torch_memory(fa_esm_forward))
            fa_esm_time_fold.append(benchmark_inference_time(fa_esm_forward))

        esm_memory_usage.append(np.mean(esm_memory_fold))
        fa_esm_memory_usage.append(np.mean(fa_esm_memory_fold))
        esm_inference_times.append(np.mean(esm_time_fold))
        fa_esm_inference_times.append(np.mean(fa_esm_time_fold))

        print(
            f"Seq Len: {seq_length}, Avg ESMC Mem: {esm_memory_usage[-1]:.3f} GB, Avg FAESMC Mem: {fa_esm_memory_usage[-1]:.3f} GB"
        )
        print(
            f"Seq Len: {seq_length}, Avg ESMC Time: {esm_inference_times[-1]:.3f} s, Avg FAESMC Time: {fa_esm_inference_times[-1]:.3f} s"
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

    # Memory Benchmark Plot
    ax1 = axes[0]
    ax1.plot(
        max_seq_lengths,
        esm_memory_usage,
        label="ESMC Memory Usage (GB)",
        marker="o",
        color=color_palette[0],
    )
    ax1.plot(
        max_seq_lengths,
        fa_esm_memory_usage,
        label="FAESMC Memory Usage (GB)",
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

    # Time Benchmark Plot
    ax2 = axes[1]
    ax2.plot(
        max_seq_lengths_filtered,
        esm_inference_times,
        label="ESMC Inference Time (s)",
        marker="o",
        color=color_palette[0],
    )
    ax2.plot(
        max_seq_lengths_filtered,
        fa_esm_inference_times,
        label="FAESMC Inference Time (s)",
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
        f"Model: ESMC\nBatch Size: {batch_size}, Data Type: {dtype}, Averaged over {repeats} runs",
        fontsize=20,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("benchmark_esmc.png", dpi=300)  # High resolution
    plt.close()

    for seq_length, fa_mem, esm_mem, fa_time, esm_time in zip(
        max_seq_lengths,
        fa_esm_memory_usage,
        esm_memory_usage,
        fa_esm_inference_times,
        esm_inference_times,
    ):
        assert fa_mem <= esm_mem, f"Seq {seq_length}: FAESMC {fa_mem:.3f} GB > ESMC {esm_mem:.3f} GB!"
        assert fa_time <= esm_time, f"Seq {seq_length}: FAESMC {fa_time:.3f} s > ESMC {esm_time:.3f} s!"