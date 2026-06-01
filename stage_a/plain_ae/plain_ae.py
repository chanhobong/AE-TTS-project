"""
plain_ae.py
- Plain Autoencoder for 2D axial slices.
- Same architecture as DiffAE (diffae-master/BeatGANs) but WITHOUT diffusion.
- Encoder: BeatGANsEncoderModel (identical to diffae-master)
- Decoder: Same output path structure, cond-only (no time embedding), from latent
"""

from __future__ import annotations

import os
import sys

# Add stageA shared diffae (BeatGANs encoder)
_plain_ae_dir = os.path.dirname(os.path.abspath(__file__))
_stage_a_root = os.path.dirname(_plain_ae_dir)
_diffae_root = os.path.join(_stage_a_root, "shared", "diffae")
if _diffae_root not in sys.path:
    sys.path.insert(0, _diffae_root)

import torch
import torch.nn as nn

# Import BeatGANs encoder from diffae-master
from model.unet import BeatGANsEncoderConfig, BeatGANsEncoderModel


def _make_encoder_config(
    img_size: int = 128,
    in_channels: int = 1,
    model_channels: int = 128,
    enc_channel_mult: tuple = (1, 1, 2, 3, 4, 4),
    enc_num_res_block: int = 2,
    out_channels: int = 512,
    attention_resolutions: tuple = (16,),
    dropout: float = 0.1,
    pool: str = "adaptivenonzero",
) -> BeatGANsEncoderConfig:
    """Build encoder config matching ffhq128_autoenc_base (CT-adapted)."""
    return BeatGANsEncoderConfig(
        image_size=img_size,
        in_channels=in_channels,
        model_channels=model_channels,
        out_hid_channels=out_channels,
        out_channels=out_channels,
        num_res_blocks=enc_num_res_block,
        attention_resolutions=attention_resolutions or (16,),
        dropout=dropout,
        channel_mult=enc_channel_mult,
        use_time_condition=False,
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        resblock_updown=True,
        use_new_attention_order=False,
        pool=pool,
    )


def plain_ae_ffhq128_ct(
    in_channels: int = 1,
    out_channels: int = 1,
    style_ch: int = 512,
) -> PlainAE:
    """Factory: Plain AE matching train_diffae_stageA_ct config (ffhq128, CT)."""
    return PlainAE(
        img_size=128,
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=128,
        channel_mult=(1, 1, 2, 3, 4),
        enc_channel_mult=(1, 1, 2, 3, 4, 4),
        enc_num_res_block=2,
        num_res_blocks=2,
        style_ch=style_ch,
        attention_resolutions=(16,),
        dropout=0.1,
    )


class PlainDecoder2D(nn.Module):
    """
    Decoder matching BeatGANs output path structure (ffhq128).
    Bottleneck 8x8, channels: 512 -> 384 -> 256 -> 128 -> 128 -> out_ch.
    Cond-only (style), no time embedding.
    """

    def __init__(
        self,
        cond_dim: int = 512,
        out_ch: int = 1,
        model_channels: int = 128,
        channel_mult: tuple = (1, 1, 2, 3, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16,),
        dropout: float = 0.1,
    ):
        super().__init__()
        # ffhq128: bottleneck 8x8, ch = 128*4 = 512
        bottleneck_ch = int(model_channels * channel_mult[-1])
        bottleneck_res = 8  # 128 -> 64 -> 32 -> 16 -> 8

        self.latent_proj = nn.Sequential(
            nn.Linear(cond_dim, bottleneck_ch * bottleneck_res * bottleneck_res),
            nn.SiLU(inplace=True),
        )
        self.bottleneck_ch = bottleneck_ch
        self.bottleneck_res = bottleneck_res

        # Up path: same structure as BeatGANs output_blocks
        # Level 4 (8x8): 512 -> 384, upsample to 16
        # Level 3 (16x16): 384 -> 256, upsample to 32
        # Level 2 (32x32): 256 -> 128, upsample to 64
        # Level 1 (64x64): 128 -> 128, upsample to 128
        # Level 0 (128x128): 128 -> out_ch
        mults = list(channel_mult)[::-1]
        ch = bottleneck_ch
        res = bottleneck_res

        self.blocks = nn.ModuleList()
        for i, mult in enumerate(mults):
            out_ch_level = int(model_channels * mult)
            use_attn = res in attention_resolutions
            for _ in range(num_res_blocks + 1):
                self.blocks.append(
                    ResBlockCondOnly(ch, out_ch_level, cond_dim, dropout=dropout)
                )
                ch = out_ch_level
                if use_attn:
                    self.blocks.append(AttentionBlock(ch))
            if i < len(mults) - 1:
                self.blocks.append(UpsampleBlock(ch))
                res *= 2

        self.out_norm = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.out_conv = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1)
        self.input_ch = ch

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        B = cond.shape[0]
        h = self.latent_proj(cond)
        h = h.view(B, self.bottleneck_ch, self.bottleneck_res, self.bottleneck_res)

        for blk in self.blocks:
            if isinstance(blk, ResBlockCondOnly):
                h = blk(h, cond)
            elif isinstance(blk, AttentionBlock):
                h = blk(h)
            else:
                h = blk(h)

        h = nn.functional.silu(self.out_norm(h))
        return self.out_conv(h)


class ResBlockCondOnly(nn.Module):
    """ResBlock with style conditioning only (BeatGANs-style scale/shift)."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(min(8, in_ch), in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.cond_proj = nn.Sequential(
            nn.SiLU(),  # Must not use inplace: cond is reused across blocks
            nn.Linear(cond_dim, out_ch),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        c = self.cond_proj(cond)
        while len(c.shape) < len(h.shape):
            c = c[..., None]
        h = h * (1 + c)
        h = self.out_layers(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block (simplified from BeatGANs)."""

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, -1)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = (C // self.num_heads) ** -0.25
        attn = torch.einsum("bhdn,bhdm->bhnm", q * scale, k * scale).softmax(dim=-1)
        h = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        h = h.reshape(B, C, H, W)
        return x + self.proj_out(h)


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class PlainAE(nn.Module):
    """
    Plain Autoencoder using DiffAE (BeatGANs) encoder, direct decoder.
    Same encoder as diffae-master; decoder reconstructs from latent without diffusion.
    """

    def __init__(
        self,
        img_size: int = 128,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 128,
        channel_mult: tuple = (1, 1, 2, 3, 4),
        enc_channel_mult: tuple = (1, 1, 2, 3, 4, 4),
        enc_num_res_block: int = 2,
        num_res_blocks: int = 2,
        style_ch: int = 512,
        attention_resolutions: tuple = (16,),
        dropout: float = 0.1,
    ):
        super().__init__()
        enc_conf = _make_encoder_config(
            img_size=img_size,
            in_channels=in_channels,
            model_channels=model_channels,
            enc_channel_mult=enc_channel_mult,
            enc_num_res_block=enc_num_res_block,
            out_channels=style_ch,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
        )
        self.encoder = BeatGANsEncoderModel(enc_conf)
        self.decoder = PlainDecoder2D(
            cond_dim=style_ch,
            out_ch=out_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward(x)

    def forward(
        self,
        x0: torch.Tensor,
        return_cond: bool = False,
    ):
        """
        Args:
            x0: (B, 1, H, W) clean slice
            return_cond: if True, also return cond (style vector)
        """
        cond = self.encoder(x0)
        x0_hat = self.decoder(cond)
        if return_cond:
            return x0_hat, cond
        return x0_hat
