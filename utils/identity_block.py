"""Identity blocks for replacing pruned transformer components.

Each identity block mimics the interface of a specific transformer component
(full block, attention, feed-forward, normalization) but passes inputs through
unchanged, effectively removing the component's computation at zero cost.
"""

from typing import Any, Optional, Tuple, Union

import torch
from diffusers.models.normalization import FP32LayerNorm


class Identity_Block_SD(torch.nn.Module):
    """Identity replacement for an entire SD3 ``JointTransformerBlock``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, hidden_states, encoder_hidden_states, temb, joint_attention_kwargs, **kwargs):
        return encoder_hidden_states, hidden_states


class Identity_Block_Flux(torch.nn.Module):
    """Identity replacement for a Flux dual-stream ``FluxTransformerBlock``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb,
        **kwargs,
    ):
        return encoder_hidden_states, hidden_states


class Identity_Block_Flux_Single(torch.nn.Module):
    """Identity replacement for a Flux single-stream ``FluxSingleTransformerBlock``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, hidden_states, temb, image_rotary_emb, **kwargs):
        return hidden_states


class Identity_Block_JointAttn(torch.nn.Module):
    """Identity replacement for a joint attention module for both SD3 and Flux."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class Identity_Block_FeedForward(torch.nn.Module):
    """Identity replacement for a feed-forward network for both SD3 and Flux."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return hidden_states


class Identity_Block_AdaLayerNormZero(torch.nn.Module):
    """Identity replacement for AdaLN-Zero normalization for SD3 and Flux's dual-stream blocks."""

    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        num_embeddings: Optional[int] = None,
        norm_type="layer_norm",
        bias=True,
        use_dual_attention=False,
        context_pre_only=False,
    ):
        super().__init__()
        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only
        if embedding_dim is not None:
            if norm_type == "layer_norm":
                self.norm = torch.nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
            elif norm_type == "fp32_layer_norm":
                self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
            else:
                raise ValueError(
                    f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
                )
        else:
            self.norm = None

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        # dim of x = [Batch, C, D]
        # dim of shift, scale, gate = [Batch, D]
        gate_msa = gate_mlp = torch.ones([x.shape[0], x.shape[-1]], dtype=x.dtype, device=x.device)
        shift_mlp = scale_mlp = torch.zeros(
            [x.shape[0], x.shape[-1]], dtype=x.dtype, device=x.device
        )
        if self.norm is not None:
            x = self.norm(x)

        if self.use_dual_attention:
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp, x, gate_msa
        else:
            if not self.context_pre_only:
                return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
            else:
                return x


class Identity_Block_AdaLayerNormZeroSingle(torch.nn.Module):
    """Identity replacement for AdaLN-Zero normalization for Flux's single-stream blocks."""

    def __init__(self, embedding_dim: Optional[int] = None, norm_type="layer_norm", bias=True):
        super().__init__()
        if embedding_dim is not None:
            if norm_type == "layer_norm":
                self.norm = torch.nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
            else:
                raise ValueError(
                    f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
                )
        else:
            self.norm = None

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_msa = torch.ones([x.shape[0], x.shape[-1]], dtype=x.dtype, device=x.device)
        if self.norm is not None:
            x = self.norm(x)

        return x, gate_msa


def get_all_identity_blocks():
    """Return a tuple of all identity block classes for isinstance checks."""
    return (
        Identity_Block_SD,
        Identity_Block_Flux,
        Identity_Block_Flux_Single,
        Identity_Block_JointAttn,
        Identity_Block_FeedForward,
        Identity_Block_AdaLayerNormZero,
        Identity_Block_AdaLayerNormZeroSingle,
    )
