"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch
import torch.nn.functional as F

from x_transformers.x_transformers import RotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
from transformers.models.llama import LlamaConfig
from torch.utils.checkpoint import checkpoint

from diffrhythm.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)
# from liger_kernel.transformers import apply_liger_kernel_to_llama
# apply_liger_kernel_to_llama()

# Text embedding


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        #text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        #text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        #text = F.pad(text, (0, seq_len - text_len), value=0)

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, cond_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim + cond_dim * 2, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], style_emb, time_emb, drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        style_emb = style_emb.unsqueeze(1).repeat(1, x.shape[1], 1)
        time_emb = time_emb.unsqueeze(1).repeat(1, x.shape[1], 1)
        # print(x.shape, cond.shape, text_embed.shape, style_emb.shape, time_emb.shape)
        x = self.proj(torch.cat((x, cond, text_embed, style_emb, time_emb), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        conv_layers=0,
        long_skip_connection=False,
        use_style_prompt=False
    ):
        super().__init__()

        cond_dim = 512
        self.time_embed = TimestepEmbedding(cond_dim)
        self.start_time_embed = TimestepEmbedding(cond_dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim, cond_dim=cond_dim)

        #self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        #self.transformer_blocks = nn.ModuleList(
        #    [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout, use_style_prompt=use_style_prompt) for _ in range(depth)]
        #)
        llama_config = LlamaConfig(hidden_size=dim, intermediate_size=dim * ff_mult, hidden_act='silu')
        llama_config._attn_implementation = 'sdpa'
        #llama_config._attn_implementation = ''
        self.transformer_blocks = nn.ModuleList(
            [LlamaDecoderLayer(llama_config, layer_idx=i) for i in range(depth)]
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=llama_config)
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.text_fusion_linears = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cond_dim, dim),
                    nn.SiLU()
                ) for i in range(depth // 2)
            ]
        )
        for layer in self.text_fusion_linears:
            for p in layer.parameters():
                p.detach().zero_()

        self.norm_out = AdaLayerNormZero_Final(dim, cond_dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        # if use_style_prompt:
        #     self.prompt_rnn = nn.LSTM(64, cond_dim, 1, batch_first=True)

    def forward_timestep_invariant(self, text, seq_len, drop_text, start_time):
        s_t = self.start_time_embed(start_time)
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
        text_residuals = []
        for layer in self.text_fusion_linears:
            text_residual = layer(text_embed)
            text_residuals.append(text_residual)
        return s_t, text_embed, text_residuals


    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        text_embed: int["b nt"],  # text  # noqa: F722
        text_residuals,
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_prompt=False,
        style_prompt=None, # [b d t]
        start_time=None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        c = t + start_time

        if drop_prompt:
            style_prompt = torch.zeros_like(style_prompt)
        
        style_embed = style_prompt # [b, 512]

        x = self.input_embed(x, cond, text_embed, style_embed, c, drop_audio_cond=drop_audio_cond)

        if self.long_skip_connection is not None:
            residual = x

        pos_ids = torch.arange(x.shape[1], device=x.device)
        pos_ids = pos_ids.unsqueeze(0).repeat(x.shape[0], 1)
        rotary_embed = self.rotary_emb(x, pos_ids)

        for i, block in enumerate(self.transformer_blocks):
            x, *_ = block(x, position_embeddings=rotary_embed)
            if i < self.depth // 2:
                x = x + text_residuals[i]

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, c)
        output = self.proj_out(x)

        return output
