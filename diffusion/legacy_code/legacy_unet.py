import math
import numpy as np

import typing

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm, trange
from abc import abstractmethod

from ..script_util import ModelType, VarType
from ..vgg5 import VGG5



def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    Se https://arxiv.org/abs/1901.09321v2
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(timesteps: torch.Tensor, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


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


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    nn.Upsample
    
    def __init__(self, channels, use_conv, *, out_channels=None, mode='nearest'):
        assert use_conv or not out_channels, "Only specify out channels when using convolutions"
        super().__init__()
        self.channels = channels
        # set out_channels as out_channels if not None else as channels
        self.out_channels = out_channels or channels
        self.mode = mode
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=(3,3), padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        # TODO: would be cheaper if interpolating after convolution
        x = F.interpolate(x, scale_factor=2, mode=self.mode)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, *, out_channels=None):
        assert use_conv or not out_channels, "Only specify out channels when using convolutions"
        super().__init__()
        self.channels = channels
        # set out_channels as out_channels if not None else as channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, kernel_size=(3,3), stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        *,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        up=False,
        down=False,
        group_norm_groups=None
    ):
        assert not (up and down), "cannot be both an up and down block"
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        # set out_channels as out_channels if not None else as channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        
        min_channels = min(self.channels, self.out_channels)
        assert ((group_norm_groups is None and min_channels%2 == 0) or
                (group_norm_groups and min_channels%group_norm_groups == 0)), \
                    "Error in assigning groups for group norm"
                    
        self.group_norm_groups = min_channels // 2 if group_norm_groups is None else group_norm_groups

        self.in_layers = nn.Sequential(
            nn.GroupNorm(self.group_norm_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, kernel_size=(3,3), padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(self.group_norm_groups, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(3,3), padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, kernel_size=(3,3), padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, kernel_size=(1,1))

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            # if scaled, apply the scaling before the convolution and to the skip
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
    


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class UNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        model_channels,
        out_channels,
        model_type: ModelType,
        var_type: VarType,
        num_res_blocks,
        dropout=0.,
        channel_mult=(1, 2, 4, 8),
        img_size=64,
        resample_mode=None,
        num_classes=None,
        use_scale_shift_norm=False,
        group_norm_groups=None
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = 2 * out_channels if var_type is VarType.learned else out_channels
        self.model_type = model_type
        self.var_type = var_type
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.img_size = img_size
        self.resample_mode = resample_mode
        self.num_classes = num_classes
        self.use_scale_shift_norm = use_scale_shift_norm
        
        assert ((group_norm_groups is None and model_channels%2 == 0) or
                (group_norm_groups and model_channels%group_norm_groups == 0)), \
                    "Error in assigning groups for group norm"
        self.group_norm_groups = model_channels // 2 if group_norm_groups is None else group_norm_groups
        
        assert ((channel_mult[0] == 1) and 
                np.all(i ** 2 == m for i, m in enumerate(channel_mult))), \
                f"{channel_mult=} not initialized correctly"
        
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
                layer = list()
                channels = mult * model_channels
                channel_width = img_size // mult
                # if first in level not the highest level
                if i == 0 and level != 0:
                    # one for downsize and one for doubling channels
                    layer.extend([
                        ResBlock(channels//2, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm, down=True),
                        ResBlock(channels//2, time_embed_dim, dropout, out_channels=channels, use_scale_shift_norm=use_scale_shift_norm)
                    ])
                layer.extend([
                    ResBlock(channels, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
                    SelfAttention(channels, channel_width)
                ])
                self.input_blocks.append(
                    TimestepEmbedSequential(*layer)
                )
        
        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
            SelfAttention(channels, channel_width),
            ResBlock(channels, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm)
        )
        
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks):
                channels = mult * model_channels
                channel_width = img_size // mult
                layer = [
                    ResBlock(2 * channels, time_embed_dim, dropout, out_channels=channels, use_scale_shift_norm=use_scale_shift_norm),
                    SelfAttention(channels, channel_width)
                ]
                # if last in level not the highest level
                if i == num_res_blocks-1 and level != 0:
                    assert channels%2 == 0, f"{channels=} not devisible by 2"
                    # one for upsize and one for halving channels
                    layer.extend([
                        ResBlock(channels, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm, up=True),
                        ResBlock(channels, time_embed_dim, dropout, out_channels=channels//2, use_scale_shift_norm=use_scale_shift_norm)
                    ])
                self.output_blocks.append(
                    TimestepEmbedSequential(*layer)
                )
        
        self.out = nn.Sequential(
            nn.GroupNorm(self.group_norm_groups, channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, self.out_channels, kernel_size=(3,3), padding=1))
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # assert (y is not None) == (self.num_classes is not None), \
        #     "must specify y if and only if the model is class-conditional"

        # used for skip connections
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.model_type is ModelType.cond_embed and y is not None:
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