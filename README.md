<div align="center">

# FAPLM: A Drop-in Efficient Pytorch Implementation of Protein Language Models

</div>

**F**lash **A**ttention **P**rotein **L**anguage **M**odels (FAPLM) is an efficient PyTorch implementation of the state-of-the-art Protein language models such as [Evolutionary Scale Modeling (ESM)](https://github.com/facebookresearch/esm) family, and [Progen2](https://github.com/salesforce/progen). FAPLM is designed to be more efficient than the official implementation, which can save up to 60% of memory usage and 70% of inference time. The key features of FAPLM are:

1. **Flash Attention**: FAESM uses the [FlashAttention](https://github.com/Dao-AILab/flash-attention) implementation, by far the most efficient implementation of the self-attention mechanism.
2. **Scalar Dot-Product Attention (SDPA)**: FAESM also provides an implementation of the [PyTorch Scalar Dot-Product Attention](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html), which is a bit slower than the FlashAttention but it's compatible with most of the system and still faster than the official ESM implementation.
3. **Same Checkpoint**: FAESM is a drop-in replacement, having the same API and checkpoint as the official implementations.

<div align="center">
  <img src="assets/figs/5-Lines-of-Code-to-Get-Started!.png" alt="Figure" width="1000">
</div>

# Table of Contents

- [FAPLM: A Drop-in Efficient Pytorch Implementation of Protein Language Models](#faesm-a-drop-in-efficient-pytorch-implementation-of-esm2)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
  - [Training \[WIP\]](#training-wip)
- [Benchmarking](#benchmarking)
- [TODOs](#todos)
- [Appreciation](#appreciation)
- [Citation](#citation)

# Installation

1. Install PyTorch 1.12 and above if you haven't: `pip install pytorch`.
2. \[*Optional*\]: Install flash-attn if you want to use the flash attention implementation.

```bash
pip install flash-attn --no-build-isolation
```

<details>
<summary>Have trouble installing flash attention?</summary>

In a clean conda env, install the `cuda-toolkit 12.3.0` for compilation:
```bash
conda install nvidia/label/cuda-12.3.0::cuda-toolkit -y
```
Then install pytorch and flash attention:
```
# Install a torch version that you like.
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
```

Another workaround is docker container. You can use the official nvidia pytorch [containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) which have all the dependencies for flash attention.
</details>


3. Install FAPLM from GitHub:

```bash
# if you want to use flash attention
pip install faesm[flash_attn]
```

```bash
# if you want to forego flash attention and just use SDPA
pip install faesm
```

# Usage

## ESM2

FAESM is a drop-in replacement for the official ESM implementation. You can use the same code as you would use the official ESM implementation. For example:

```python
import torch
from faesm.esm import FAEsmForMaskedLM

# Step 1: Load the FAESM model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FAEsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device).eval().to(torch.float16)
# Step 2: Prepare a sample input sequence
sequence = "MAIVMGRWKGAR"
inputs = model.tokenizer(sequence, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
# Step 3: Run inference with the FAESM model
outputs = model(**inputs)
# Step 4: Process and print the output logits and repr.
print("Logits shape:", outputs['logits'].shape)  # (batch_size, sequence_length, num_tokens)
print("Repr shape:", outputs['last_hidden_state'].shape)  # (batch_size, sequence_length, hidden_size)
# Step 5: start the repo if the code works for u!
```

## ESM-C

Right after EvolutionaryScale release [ESM-C](https://www.evolutionaryscale.ai/blog/esm-cambrian), we follow up with the flash attention version of ESM-C in FAESM. You can run ESM-C easily with the following code:

```python
from faesm.esmc import ESMC
sequence = ['MPGWFKKAWYGLASLLSFSSFI']
model = ESMC.from_pretrained("esmc_300m",use_flash_attn=True).to("cuda")
input_ids = model.tokenizer(sequence, return_tensors="pt")["input_ids"].to("cuda")
output = model(input_ids)
print(output.sequence_logits.shape)
print(output.embeddings.shape)

```

## ProGen2

For autoregressive protein language like ProGen2.

```python
import torch
from faesm.progen2 import ProGenForCausalLM
from transformers import AutoTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Available model from HF: ["jinyuan22/ProGen2-small", "jinyuan22/ProGen2-base", "jinyuan22/ProGen2-xlarge"]
model = ProGenForCausalLM.from_pretrained("jinyuan22/ProGen2-small").to(torch.float16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained("jinyuan22/ProGen2-small")

sequence = "2GFLPFRGADM1"

inputs = tokenizer(sequence, return_tensors="pt").to(device)
logits = model(inputs.input_ids, labels=inputs.input_ids).logits[0,...]

```

### Training \[WIP\]

Working on an example training script for MLM training on Uniref50. For now, you can use the same training logic as how you would train the official ESM since the FAESM has no difference in the model architecture.
It's recommended to use the flash attention for training. Because in the forward pass, it unpads the input sequences to remove all the padding tokens, which 1) speeds up the training & reduces the memory usage and 2) it doesn't require batching sequences of similar length to avoid padding. Also, SDPA is still a good alternative if you can't install flash attention as used in [DPLM](https://github.com/bytedance/dplm).

# Benchmarking

### FAESM vs. Official ESM2

Below is the comparison of peak memory usage and inference time of FAESM with the official ESM2. We show that FAESM can save memory usage by up to 60% and inference time by up to 70% (length 1000). The benchmarking is done on ESM-650M with batch size 8, and a single A100 with 80GB of memory.

![benchmark](assets/figs/benchmark.png)

Below [@ANaka](https://github.com/ANaka) compares the SDFA implementation vs. official ESM2 (see his [PR](https://github.com/pengzhangzhi/faesm/pull/3)), where we can still get ~30% reduction by just pure pytorch, not too bad :)
![image](https://github.com/user-attachments/assets/5e9c3686-884c-4dd6-913f-d32fd158c6cd)

You can reproduce the benchmarking of ESM2 by running the following command:

```bash
pytest tests/benchmark.py
```

To test errors between FAESM and the official ESM2 implementation, you can run:

```bash
pytest tests/test_compare_esm.py
```

### FAESM-C vs. Official ESM-C

Below we show the scaling of FAESM-C with the official ESM-C, using FAESM-C we can save 60% memory usage and 70% inference time.

![benchmark_esmc](assets/figs/benchmark_esmc.png)

Run the following script to reproduce the benchmarking:

```python
pytest tests/benchmark_esmc.py
```

Run the script to test the errors between FAESM-C and the official ESM-C implementation:

```python
pytest tests/test_compare_esmc.py
```

### FAProgen2 vs. Official Progen2

Save up to 60% of memory and run time by using FAProgen2.
![benchmark_progen](assets/figs/FAProGen2_benchmark.png)

# TODOs

- Training script
- Integrate FAESM into ESMFold

# Appreciation

- The Rotary code is from [esm-efficient](https://github.com/uci-cbcl/esm-efficient).
- The ESM modules and the SDPA attention module are inspired by [ESM](https://github.com/facebookresearch/esm) and [DPLM](https://github.com/bytedance/dplm).
- I want to highlight that [esm-efficient](https://github.com/uci-cbcl/esm-efficient) also supports Flash Attention and offers more features such as quantitation and lora. Please check it out!!

This project started as a mutual disappointment with [Alex Tong(@atong01)](https://github.com/atong01) about why there is no efficient implementation of ESM (wasted a lot compute in training pLMs :(. He later helped me debugged the precision errors in my implementation and organize this repo. In the process, I talked @MuhammedHasan regarding his ESM-efficent implementation (see the issues [1](https://github.com/uci-cbcl/esm-efficient/issues/3) and [2](https://github.com/uci-cbcl/esm-efficient/issues/5)), and also Tri Tao about flash attention (see the [issue](https://github.com/Dao-AILab/flash-attention/issues/1359)). Of course shoutout to the ESM teams for creating the ESM family. None of the pieces of code would be possible without their help.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=pengzhangzhi/faesm&type=Date)](https://star-history.com/#pengzhangzhi/faesm&Date)

# License

This work is licensed under the MIT license. However, it contains altered and unaltered portions of code licensed under MIT, Apache 2.0, and Cabrian Open License Agreement.

- ESM: MIT Licensed
- DPLM: [Apache 2.0 Licensed](https://github.com/bytedance/dplm/blob/main/LICENSE)
- ESMC: [See ESM Licensing](https://github.com/evolutionaryscale/esm/blob/main/LICENSE.md) in particular the [Cambrian Open License Agreement](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement)

We thank the creators of these prior works for their contributions and open licensing. We also note that model weights may have separate licensing. The ESMC 300M Model is licensed under the EvolutionaryScale Cambrian Open License Agreement. The ESMC 600M Model is licensed under the EvolutionaryScale Cambrian Non-Commercial License Agreement.

# Citation

Please cite this repo if you use it in your work.

```bibtex
@misc{faesm2024,
  author       = {Fred Zhangzhi Peng,Pranam Chatterjee, and contributors},
  title        = {FAESM: An efficient PyTorch implementation of Evolutionary Scale Modeling (ESM)},
  year         = {2024},
  howpublished = {\url{https://github.com/pengzhangzhi/faesm}},
  note         = {Efficient PyTorch implementation of ESM with FlashAttention and Scalar Dot-Product Attention (SDPA)},
  abstract     = {FAESM is a drop-in replacement for the official ESM implementation, designed to save up to 60% memory usage and 70% inference time, while maintaining compatibility with the ESM API.},
}
```
