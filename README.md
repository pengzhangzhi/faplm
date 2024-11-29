# FAESM: An efficient pytorch implementation of ESM

# Installation

Install torch if you haven't `pip install pytorch`.

[*Optional*]: Install flash-attn if you want to use the flash attention implementation, which is the fastest and efficient implementation. However, it can be a bit tricky to install so you can skip this step without any problem. In that case, you can will use Pytorch SDPA attention which is a bit slower but still better the official ESM implementation. 

```bash

pip install flash-attn --no-build-isolation

```

Having trouble installing flash attention but still want to use it? A workaround is docker container. You can use the official nvidia pytorch containers which are well-suited for flash attention. 

```bash
```


Install FAESM from this repo.

```bash
pip install git+
```

# Usage


### Inference
The uses of FAESM is the same as the official ESM implementation, and you can load the checkpoints from huggingface facebook. For example:

```python

```

### Training [WIP]
Working on an example training script for MLM training on Uniref50. For now, you can use the same training logic as how you would train the official ESM since the FAESM has no difference in the model architecture. 

It's recommended to use the flash attention for training. Because in the forward pass, it unpads the input sequences to remove all the padding tokens, which 1) speeds up the training & reduces the memory usage and 2) it doesn't require batching sequences of similar length to avoid padding. However, SDPA is still a good alternative if you can't install flash attention. 


# Benchmarking

Below we compare the logits and representations of FAESM with the official ESM2, and show that they are numerically identical with error less than 1e-6. 


We compare the inference time and peak memory usage of FAESM with the official ESM2 with the same checkpoint , under batch size of and sequence length of . We should that FAESM is xxx times faster and xxx times more memory efficient than the official ESM2. 


You can reproduce the benchmarking by running the following command:

```bash

```
# Appreciation


- The Rotary code is inspired by [esm-efficient](https://github.com/uci-cbcl/esm-efficient)
- The ESM modules and the SDPA attention module is inspired by [ESM](https://github.com/facebookresearch/esm) and [DPLM](https://github.com/bytedance/dplm).

This project started as my mutual disappointment with Alex Tong about why there is no efficient implementation of ESM (wasted a lot compute in training pLMs :(. He later helped me debugged the precision errors in my implementation and organize this repo. In the process, I talked @MuhammedHasan regarding his ESM-efficent implementation (see the issues [1](https://github.com/uci-cbcl/esm-efficient/issues/3) and [2](https://github.com/uci-cbcl/esm-efficient/issues/5)), and also Tri Tao about flash attention (see the [issue](https://github.com/Dao-AILab/flash-attention/issues/1359)). Of course shoutout to the ESM teams for creating the ESM family. None of the piece of code would be possible without their help. 

# Citation

Please cite this repo if you use it in your work.

```bibtex   
```