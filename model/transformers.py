"""Custom forward pass for Flux single transformer blocks with identity-block support."""

import torch


def forward_FluxSingleTransformerBlock(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
):
    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    joint_attention_kwargs = joint_attention_kwargs or {}
    attn_output = self.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    gate = gate.unsqueeze(1)
    # Slicing operation based on gate.shape[2] only works when we replace proj_mlp & proj_out
    # It's because proj_mlp and proj_out expand and shrink the dimensions but when we replace them
    # with identity module, they simply returns the inputs as the outputs, leading to dimension mismatch.
    # To ensure the same dimension, we perform slicing here.
    hidden_states = gate * self.proj_out(hidden_states)[:, :, : gate.shape[2]]
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states
