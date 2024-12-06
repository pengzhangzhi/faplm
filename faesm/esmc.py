"""This file is adapted from esmc which is licensed under the EvolutionaryScale Cambrian Open
License Agreement.

This file is altered by the authors contributors to the FAESM repository.
"""
from __future__ import annotations

flash_attn_installed = True
try:
    from flash_attn import flash_attn_varlen_qkvpacked_func

    from faesm.fa_utils import RotaryEmbedding as FAEsmRotaryEmbedding
    from faesm.fa_utils import unpad
except ImportError as e:
    flash_attn_installed = False
    print(
        f"""
        [Warning] Flash Attention not installed.
        By default, we will use PyTorch SDPA attention.
        {e}
        """
    )

import contextlib
import functools
import math
import os
from functools import cache, partial
from pathlib import Path
from typing import Callable

import attr
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from attr import dataclass
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from faesm.torch_utils import RotaryEmbeddingTorch

esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")


@cache
def data_root(model: str):
    if "INFRA_PROVIDER" in os.environ:
        return Path("")
    elif model.startswith("esmc-300"):
        path = Path(snapshot_download(repo_id="EvolutionaryScale/esmc-300m-2024-12"))
    elif model.startswith("esmc-600"):
        path = Path(snapshot_download(repo_id="EvolutionaryScale/esmc-600m-2024-12"))
    else:
        raise ValueError(f"{model=} is an invalid model name.")
    return path


def ESMC_300M_202412(device: torch.device | str = "cpu", use_flash_attn=True):
    with torch.device(device):
        model = ESMC(
            d_model=960,
            n_heads=15,
            n_layers=30,
            tokenizer=esm_tokenizer,
            use_flash_attn=use_flash_attn,
        ).eval()
    state_dict = torch.load(
        data_root("esmc-300") / "data/weights/esmc_300m_2024_12_v0.pth",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    return model


def ESMC_600M_202412(device: torch.device | str = "cpu", use_flash_attn=True):
    with torch.device(device):
        model = ESMC(
            d_model=1152,
            n_heads=18,
            n_layers=36,
            tokenizer=esm_tokenizer,
            use_flash_attn=use_flash_attn,
        ).eval()
    state_dict = torch.load(
        data_root("esmc-600") / "data/weights/esmc_600m_2024_12_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)

    return model


ESMC_600M = "esmc_600m"
ESMC_300M = "esmc_300m"
LOCAL_MODEL_REGISTRY: dict[str, Callable] = {
    ESMC_600M: ESMC_600M_202412,
    ESMC_300M: ESMC_300M_202412,
}


def load_local_model(
    model_name: str, device: torch.device = torch.device("cpu"), use_flash_attn=True
) -> nn.Module:
    if model_name not in LOCAL_MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in local model registry.")
    return LOCAL_MODEL_REGISTRY[model_name](device, use_flash_attn)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbeddingTorch(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x, seq_id):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = (
            self.q_ln(query_BLD).to(query_BLD.dtype),
            self.k_ln(key_BLD).to(query_BLD.dtype),
        )
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        n_heads = self.n_heads
        reshaper = functools.partial(einops.rearrange, pattern="b s (h d) -> b h s d", h=n_heads)

        query_BHLD, key_BHLD, value_BHLD = map(reshaper, (query_BLD, key_BLD, value_BLD))

        if seq_id is not None:
            # Where True, enable participation in attention.
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask_BHLL = mask_BLL.unsqueeze(1)

            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, mask_BHLL
            )
        else:
            # Shortcut, if we don't use attention biases then torch
            # will autoselect flashattention as the implementation
            context_BHLD = F.scaled_dot_product_attention(query_BHLD, key_BHLD, value_BHLD)
        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
        return self.out_proj(context_BLD)


class FAMultiHeadAttention(MultiHeadAttention):
    def __init__(self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True):
        super().__init__(d_model, n_heads, bias, qk_layernorm)
        self.rotary = FAEsmRotaryEmbedding(d_model // n_heads, persistent=False)

    def forward(
        self,
        x,
        cu_seqlens,
        max_seqlen,
    ):
        scale = self.d_head**-0.5
        qkv_ND3 = self.layernorm_qkv(x)
        q, k, v = torch.chunk(qkv_ND3, 3, dim=-1)
        q, k = (
            self.q_ln(q).to(q.dtype),
            self.k_ln(k).to(q.dtype),
        )
        (q, k, v) = map(
            lambda x: einops.rearrange(x, "n (h d) -> n h d", h=self.n_heads), (q, k, v)
        )
        qkv_N3HD = torch.stack((q, k, v), dim=1)
        qkv_N3HD = self.rotary(qkv=qkv_N3HD, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        out = flash_attn_varlen_qkvpacked_func(
            qkv_N3HD, cu_seqlens, max_seqlen, softmax_scale=scale
        )
        out = einops.rearrange(out, "n h d -> n (h d)")
        return self.out_proj(out)


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.

    This module splits the input tensor along the last dimension and applies the SiLU (Swish)
    activation function to the first half, then multiplies it by the second half.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


def gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    hidden_dim = int(expansion_ratio * d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim, bias=bias),
        nn.GELU(),
        nn.Linear(hidden_dim, d_model, bias=bias),
    )


class UnifiedTransformerBlock(nn.Module):
    """A unified transformer block that can optionally incorporate geometric attention.

    This class defines a transformer block that can be configured to use geometric attention
    alongside the standard multi-head attention mechanism. It is designed to be a flexible
    component of transformer-based models, allowing for the integration of geometric reasoning.

    Parameters
    ----------
    d_model : int
        The dimensionality of the input and output features of the transformer block.
    n_heads : int
        The number of attention heads in the multi-head attention mechanism.
    n_layers : int
        The number of layers in the transformer block.
    use_geom_attn : bool, optional
        Whether to use geometric attention in addition to the standard multi-head attention. Defaults to False.
    v_heads : int, optional
        The number of heads to use for the geometric attention mechanism, if enabled. Must be specified if `use_geom_attn` is True.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_geom_attn: bool = False,
        use_flash_attn: bool = True,
        v_heads: int | None = None,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1,
        mask_and_zero_frameless: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",  # swiglu | gelu
    ):
        super().__init__()
        self.use_flash_attn = use_flash_attn and flash_attn_installed
        if self.use_flash_attn:
            self.attn = FAMultiHeadAttention(d_model, n_heads, bias, qk_layernorm=qk_layernorm)
        else:
            self.attn = MultiHeadAttention(d_model, n_heads, bias, qk_layernorm=qk_layernorm)
        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        self.scaling_factor = residue_scaling_factor

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Forward pass for the UnifiedTransformerBlock.

        Parameters
        ----------
        x : torch.Tensor[float]
            Input tensor to the transformer block, typically the output from the previous layer.
        sequence_id : torch.Tensor[int]
            Tensor containing sequence IDs for each element in the batch, used for attention masking.
        Returns
        -------
        torch.Tensor[float]
            The output tensor after applying the transformer block operations.
        """
        if self.use_flash_attn:
            r1 = self.attn(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        else:
            r1 = self.attn(x, sequence_id)
        x = x + r1 / self.scaling_factor
        r3 = self.ffn(x) / self.scaling_factor
        x = x + r3

        return x


class TransformerStack(nn.Module):
    """A stack of transformer blocks used in the ESM-3 model. Each block is a
    UnifiedTransformerBlock, which can either be geometric attention or standard multi-head
    attention.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads.
        v_heads (int): The number of voting heads.
        n_layers (int): The number of transformer blocks in the stack.
        n_layers_geom (int, optional): The number of transformer blocks that use geometric attention.
        scale_residue (bool, optional): Whether to scale the residue connections in each transformer block.
        mask_and_zero_frameless (bool, optional): Whether to mask and zero frameless positions in the input.
            Only applies in the geometric attention blocks, which is conditioned on the structure
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: int | None,
        n_layers: int,
        n_layers_geom: int = 1,
        scale_residue: bool = True,
        mask_and_zero_frameless: bool = False,
        bias: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",  # swiglu | gelu
        expansion_ratio: float = 8 / 3,
        use_flash_attn=True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    v_heads=v_heads,
                    use_geom_attn=i < n_layers_geom,
                    residue_scaling_factor=(math.sqrt(n_layers / 36) if scale_residue else 1.0),
                    expansion_ratio=expansion_ratio,
                    mask_and_zero_frameless=mask_and_zero_frameless,
                    bias=bias,
                    qk_layernorm=qk_layernorm,
                    ffn_type=ffn_type,
                    use_flash_attn=use_flash_attn,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the TransformerStack.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).
            sequence_id (torch.Tensor): The sequence ID tensor of shape (batch_size, sequence_length).
        Returns:
            post_norm: The output tensor of shape (batch_size, sequence_length, d_model).
            pre_norm: The embedding of shape (batch_size, sequence_length, d_model).
        """
        for block in self.blocks:
            x = block(x, sequence_id, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.norm(x), x


def RegressionHead(d_model: int, output_dim: int, hidden_dim: int | None = None) -> nn.Module:
    """Single-hidden layer MLP for supervised output.

    Args:
        d_model: input dimension
        output_dim: dimensionality of the output.
        hidden_dim: optional dimension of hidden layer, defaults to d_model.
    Returns:
        output MLP module.
    """
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class ESMCOutput:
    sequence_logits: torch.Tensor
    embeddings: torch.Tensor | None


class ESMC(nn.Module):
    """ESMC model implementation.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads in the transformer layers.
        n_layers (int): The number of transformer layers.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, tokenizer, use_flash_attn=True):
        super().__init__()
        self.use_flash_attn = use_flash_attn and flash_attn_installed
        self.embed = nn.Embedding(64, d_model)
        self.transformer = TransformerStack(
            d_model, n_heads, None, n_layers, n_layers_geom=0, use_flash_attn=self.use_flash_attn
        )
        self.sequence_head = RegressionHead(d_model, 64)
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls, model_name: str = ESMC_600M, device: torch.device | None = None, use_flash_attn=True
    ) -> ESMC:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_local_model(model_name, device=device, use_flash_attn=use_flash_attn)
        if device.type != "cpu":
            model = model.to(torch.bfloat16)
        assert isinstance(model, ESMC)
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def raw_model(self):
        return self

    def forward(
        self,
        sequence_tokens: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        pad_fn: callable | None = None,
    ) -> ESMCOutput:
        """Performs forward pass through the ESMC model. Check utils to see how to tokenize inputs
        from raw data.

        Args:
            sequence_tokens (torch.Tensor, optional): The amino acid tokens.
            sequence_id (torch.Tensor, optional): The sequence ID.

        Returns:
            ESMCOutput: The output of the ESMC model.
        """
        sequence_id = sequence_tokens == self.tokenizer.pad_token_id
        if self.use_flash_attn:
            sequence_tokens, cu_seqlens, max_seqlen, _, pad_fn = unpad(
                sequence_tokens.unsqueeze(-1), ~sequence_id
            )
            sequence_tokens = sequence_tokens.squeeze(-1)
            sequence_id = None

        else:
            pad_fn = lambda x: x
            cu_seqlens = None
            max_seqlen = None

        x = self.embed(sequence_tokens)
        x, _ = self.transformer(
            x, sequence_id=sequence_id, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
        sequence_logits = self.sequence_head(x)
        sequence_logits = pad_fn(sequence_logits)
        output = ESMCOutput(sequence_logits=sequence_logits, embeddings=x)
        return output


if __name__ == "__main__":
    sequence = [
        "MPGWFKKAWYGLASLLSFSSFILIIVALVVPHWLSGKILCQTGVDLVNATDRELVKFIGDIYYGLFRGCKVRQCGLGGRQSQFTIFPHLVKELNAGLHVMILLLLFLALALALVSMGFAILNMIQVPYRAVSGPGGICLWNVLAGGVVALAIASFVAAVKFHDLTERIANFQEKLFQFVVVEEQYEESFWICVASASAHAANLVVVAISQIPLPEIKTKIEEATVTAEDILY"
    ]
    model = ESMC.from_pretrained("esmc_300m", use_flash_attn=True).to("cuda")
    input_ids = model.tokenizer(sequence, return_tensors="pt")["input_ids"].to("cuda")
    output = model(input_ids)
    print(output.sequence_logits.mean())
    print(output.embeddings.mean())
    print(output.embeddings.max())
