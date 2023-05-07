import math
import numpy as np

import typing

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm, trange
from abc import abstractmethod

from .script_util import ModelType, VarType
from .vgg5 import VGG5


def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    Se https://arxiv.org/abs/1901.09321v2
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class SimpleUp(nn.Module):
    def __init__(self, *, scale: int=2, mode: str='bilinear'):
        super().__init__()
        assert mode in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-exact'], \
            f"Select valid mode for upsampler, {mode} not recognised"
        self.scale = scale
        self.mode = mode
        
    def forward(self, x: torch.Tensor):
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode)


class SimpleDown(nn.Module):
    def __init__(self, *, scale: int=2):
        super().__init__()
        
        self.scale = scale
        self.avgpool = nn.AvgPool2d(kernel_size=(scale, scale), stride=(scale, scale))
        
    def forward(self, x: torch.Tensor):
        return self.avgpool(x)


class SimpleResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        *,
        out_channels=None,
        dropout,
        group_norm_groups,
    ):
        
        super().__init__()
        
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.dropout = dropout
        self.group_norm_groups = group_norm_groups
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(self.group_norm_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, kernel_size=(3,3), padding=1),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(self.group_norm_groups, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(3,3), padding=1)),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )
        
        # select skip connection type
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, kernel_size=(1,1))
    
    def forward(self, x, emb):
        # get skip
        skip = self.skip_connection(x)
        
        # get timestep embedding and format
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        
        x = self.in_layers(x)
        x = x + emb_out
        x = self.out_layers(x)
        return x + skip


class SimpleAttention(nn.Module):
    """
    Multi-Head Attention from Attention Is All You Need but without projecting the
    query, key, value into a emb dim (multiplying with matrix)
    Just splitting into num_heads * batchdim and doing attention on each channel
    """
    def __init__(self, channels, num_heads, group_norm_groups):
        
        super().__init__()
        
        assert channels % num_heads == 0, f"{channels=} has to be devisible by {num_heads=}"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.group_norm_groups = group_norm_groups
        
        self.qkv = nn.Sequential(
            nn.GroupNorm(group_norm_groups, channels),
            nn.Conv1d(channels, 3 * channels, kernel_size=1)
        )
        self.out = zero_module(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1) # flatten image to vector
        qkv = self.qkv(x) # project x to query, key, value
        attn = self.mha(qkv) # scaled dot product attention
        attn = self.out(attn) # linear combination of channels
        return (x + attn).reshape(b, c, *spatial) # reshape to get image again

    def mha(self, qkv: torch.Tensor):
        """
        Splits the channel dim into multi heads before doing scaled dot product attention
        """
        batch_dim, _, length = qkv.shape
        # split into heads and then unpack q, k, v
        q, k, v = qkv.reshape(batch_dim * self.num_heads, self.head_channels * 3, length).split(self.head_channels, dim=1)
        # // scaled dot product attention
        scale = 1 / math.sqrt(math.sqrt(length)) # length or self.head_channels
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(batch_dim, -1, length)


class SimpleUNet(nn.Module):
    def __init__(self,
        in_channels,
        model_channels,
        num_res_blocks,
        model_type: ModelType,
        var_type: VarType,
        *,
        out_channels=None,
        dropout=0.,
        num_model_levels=3,
        img_size=64,
        num_classes=None,
        group_norm_groups=None,
        time_embed_max_period=10_000,
        num_attention_heads=4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.model_type = model_type
        self.var_type = var_type
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.num_model_levels = num_model_levels
        self.img_size = img_size
        self.num_classes = num_classes
        self.time_embed_max_period = time_embed_max_period
        self.num_attention_heads = num_attention_heads
        
        self.channel_mult = channel_mult = [2 ** i for i in range(num_model_levels)]
        out_channels = out_channels or in_channels
        self.out_channels = 2 * out_channels if var_type is VarType.learned else out_channels # doubles num out channels when having to learn variance
        
        if group_norm_groups is None:
            assert model_channels % 2 == 0, "Select num_groups or change model_channels to be even"
            self.group_norm_groups = group_norm_groups = model_channels // 2
        else:
            assert model_channels % group_norm_groups == 0, f"{model_channels=} has to be divisible by {group_norm_groups=}"
            self.group_norm_groups = group_norm_groups
        assert np.all(img_size%m == 0 for m in channel_mult), \
                f"{img_size=} not divisible with all {channel_mult=}"
        assert ((model_type is not ModelType.cond_embed) or
                (num_classes is not None)), \
                "cannot use conditional embedding without selecting num_classes"
        
        time_embed_dim = 4 * self.model_channels
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, model_channels, kernel_size=(3,3), padding=1)
            )
        ])
        for level, mult in enumerate(channel_mult):
            # n blocks of resblock and self attention
            for i in range(num_res_blocks):
                channels = mult * model_channels
                # if first in level not the highest level
                if i == 0 and level != 0:
                    layer = [
                        SimpleDown(),
                        SimpleResBlock(channels // 2, time_embed_dim, out_channels=channels, dropout=dropout, group_norm_groups=group_norm_groups),
                        SimpleAttention(channels, num_attention_heads, group_norm_groups)
                    ]
                else:
                    layer = [
                        SimpleResBlock(channels, time_embed_dim, dropout=dropout, group_norm_groups=group_norm_groups),
                        SimpleAttention(channels, num_attention_heads, group_norm_groups)
                    ]
                self.input_blocks.append(
                    TimestepEmbedSequential(*layer)
                )
        
        channels = channel_mult[-1] * model_channels
        self.middle_block = TimestepEmbedSequential(
            SimpleResBlock(channels, time_embed_dim, dropout=dropout, group_norm_groups=group_norm_groups),
            SimpleAttention(channels, num_attention_heads, group_norm_groups),
            SimpleResBlock(channels, time_embed_dim, dropout=dropout, group_norm_groups=group_norm_groups)
        )
        
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks):
                channels = mult * model_channels
                # if not last in level and not the highest level
                if i != num_res_blocks-1 or level == 0:
                    layer = [
                        SimpleResBlock(2*channels, time_embed_dim, out_channels=channels, dropout=dropout, group_norm_groups=group_norm_groups),
                        SimpleAttention(channels, num_attention_heads, group_norm_groups)
                    ]
                else:
                    layer = [
                        SimpleResBlock(2*channels, time_embed_dim, out_channels=channels//2, dropout=dropout, group_norm_groups=group_norm_groups),
                        SimpleAttention(channels//2, num_attention_heads, group_norm_groups),
                        SimpleUp()
                    ]
                self.output_blocks.append(
                    TimestepEmbedSequential(*layer)
                )
        
        self.out = nn.Sequential(
            nn.GroupNorm(self.group_norm_groups, model_channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, self.out_channels, kernel_size=(3,3), padding=1))
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor=None):
        hs = [] # saves skip connections
        
        emb = self.timestep_embedding(timesteps)
        emb = self.time_embed(emb)

        if (self.model_type is ModelType.cond_embed) and (y is not None):
            assert y.shape == (x.shape[0],), "Num target classes not equal bach size"
            emb = emb + self.label_emb(y)

        h = x.type(torch.float32)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)
    
    def timestep_embedding(self, timesteps: torch.Tensor):
        """
        Create sinusoidal timestep embeddings.
        """
        half = self.model_channels // 2
        freqs = torch.exp(
            -math.log(self.time_embed_max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.model_channels % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding