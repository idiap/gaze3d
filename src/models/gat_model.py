# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from functools import partial
from typing import List

import einops
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class Swin3D(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.encoder = torch.hub.load(
            "facebookresearch/omnivore", model="omnivore_swinT", pretrained=pretrained
        )
        self.encoder = self.encoder.trunk
        self.output_features = self.encoder.num_features

    def forward(self, x):
        return self.encoder(x, ["stage3"])[0]


class MLPHead(nn.Module):

    """
    Very simple MLP head with a fixed number of layers and hidden dimensions.
    """

    def __init__(self, in_features, hidden_dim, num_layers, out_features, dropout=0):
        super().__init__()

        self.in_features = in_features
        self.out_dimension = hidden_dim

        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([in_features] + h, h + [out_features])
        )
        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape is (batch_size, seq_len, in_features)
        b, s, f = x.size()

        x = rearrange(x, "b s f -> (b s) f")
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if self.dropout and i < self.num_layers:
                x = self.dropout(x)
        x = rearrange(x, "(b s) f -> b s f", b=b, s=s)

        return x


class HeadDict:
    def __init__(self, names: List[str], modules: List[partial]):
        self.heads = dict(zip(names, modules))


class GaT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        head_dict: HeadDict,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.ModuleDict(
            {
                name: module(in_features=self.encoder.output_features)
                for name, module in head_dict.heads.items()
            }
        )

    def forward(self, x: torch.Tensor):
        # x: B x T x C x H x W
        b, t_in, c, h, w = x.size()
        assert x.dim() == 5
        x = einops.rearrange(x, "b t c h w -> b c t h w")

        # If the input is an image, we duplicate the image to have a temporal dimension of 2
        if t_in == 1:
            x = x.repeat(1, 1, 2, 1, 1)
            t = 2
        else:
            t = t_in

        # Encode Image or Video
        x = self.encoder(x)  # B x num_features x T/2 x H/32 x W/32

        # Upsample temporal dimension to match the original temporal dimension
        if t_in != 1:
            x = F.interpolate(
                x, size=t_in, mode="trilinear", align_corners=True
            )  # B x num_features x T

        # Aggreate the spatial dimension
        x = torch.mean(x, [-2, -1])  # B x num_features x T/2
        x = rearrange(x, "b f t -> b t f")  # B x T x num_features

        # Decoder
        out_prediction = {}
        for name, module in self.decoder.items():
            out_prediction[name] = module(x)

        return out_prediction
