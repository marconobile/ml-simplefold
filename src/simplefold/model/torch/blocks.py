#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
from torch import nn
from timm.models.vision_transformer import Mlp
from model.torch.layers import modulate, SwiGLUFeedForward


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        self_attention_layer,
        hidden_size,
        mlp_ratio=4.0,
        use_swiglu=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = self_attention_layer()
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)
        else:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.initialize_weights()

        self.cluster_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        torch.nn.init.normal_(self.cluster_proj.weight, mean=0.0, std=1e-6)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT encoder blocks:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        latents,
        c,
        cluster_emb=None,
        **kwargs,
    ):
        if cluster_emb  is not None:
            cluster_emb = self.cluster_proj(cluster_emb) # in_dims/out_dims: (bs, natoms, emb_dim)
            c = cluster_emb + c.unsqueeze(1) # out_dims: (bs, natoms, emb_dim)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ( # all out shapes are: ((bs, emb_dim)
                self.adaLN_modulation(c).chunk(6, dim=-1) # in_dims of c: (bs, emb_dim)
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ( # all out shapes are: ((bs, emb_dim)
                self.adaLN_modulation(c).chunk(6, dim=1) # in_dims of c: (bs, emb_dim)
            )

        _latents = self.attn(
            modulate(self.norm1(latents), shift_msa, scale_msa), # modulate broadcasts at atoms shape: out_dims: (bs, natoms, emb_dim)
            **kwargs
        )
        if _latents.shape != gate_msa.shape:
            latents = latents + gate_msa.unsqueeze(1) * _latents
            latents = latents + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(latents), shift_mlp, scale_mlp)
            )
        else:
            latents = latents + gate_msa * _latents
            latents = latents + gate_mlp * self.mlp(
                modulate(self.norm2(latents), shift_mlp, scale_mlp)
            )

        return latents


class TransformerBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        self_attention_layer,
        hidden_size,
        mlp_ratio=4.0,
        use_swiglu=False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = self_attention_layer()
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )

    def forward(
        self,
        latents,
        **kwargs,
    ):
        _latents = self.attn(self.norm1(latents), **kwargs)
        latents = latents + _latents
        latents = latents + self.mlp(self.norm2(latents))
        return latents


class HomogenTrunk(nn.Module):
    def __init__(self, block, depth):
        super().__init__()
        self.blocks = nn.ModuleList([block() for _ in range(depth)])

    def forward(self, latents, c, **kwargs):
        for i, block in enumerate(self.blocks):
            kwargs["layer_idx"] = i
            latents = block(latents=latents, c=c, **kwargs)
        return latents
