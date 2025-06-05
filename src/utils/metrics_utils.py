# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import math

import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F


def spherical2cartesial(x):
    output = torch.zeros(x.size(0), 3).to(x)
    output[:, 2] = -torch.cos(x[:, 1]) * torch.cos(x[:, 0])
    output[:, 0] = torch.cos(x[:, 1]) * torch.sin(x[:, 0])
    output[:, 1] = torch.sin(x[:, 1])

    return output


def cartesial2spherical(x):
    output = torch.zeros(x.size(0), 2).to(x)
    assert x.size(1) == 3
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-7).to(x)
    output[:, 0] = torch.atan2(x_norm[:, 0], -x_norm[:, 2])
    output[:, 1] = torch.asin(x_norm[:, 1])
    return output


def compute_angular_error(input, target):
    input_cart = spherical2cartesial(input)
    target_cart = spherical2cartesial(target)

    sim = F.cosine_similarity(input_cart, target_cart, dim=1, eps=1e-10)
    sim = F.hardtanh_(sim, min_val=-1.0, max_val=1.0)
    output_dot = torch.acos(sim).sum() * 180 / math.pi
    return output_dot


def compute_angular_error_cartesian(input, target, only_middle=True):
    if only_middle:
        t = input.size(1)
        mid_t = (
            (t // 2) if t == 1 or t % 2 != 0 else (t // 2) - 1
        )  # defined middle frame in even case
        input = input[:, mid_t : mid_t + 1, :]
        target = target[:, mid_t : mid_t + 1, :]

    # Normalize the input and target
    input_cart = torch.nn.functional.normalize(input, p=2, dim=2, eps=1e-8)
    target_cart = torch.nn.functional.normalize(target, p=2, dim=2, eps=1e-8)

    input_cart = rearrange(input_cart, "b t d -> (b t) d")
    target_cart = rearrange(target_cart, "b t d -> (b t) d")

    sim = F.cosine_similarity(input_cart, target_cart, dim=1, eps=1e-8)
    # handle the case when acos is not defined
    sim = F.hardtanh_(sim, min_val=-1 + 1e-8, max_val=1 - 1e-8)
    output_dot = torch.acos(sim).sum() * 180 / math.pi

    return output_dot
