# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import math
from typing import List

import torch
import torch.nn.functional as F
from einops import rearrange


class AngularLoss(torch.nn.Module):
    def __init__(
        self, task_name: List[str], task_weight: List[float] = None, compute_only_2d=False
    ):
        super(AngularLoss, self).__init__()

        self.compute_only_2d = compute_only_2d

        assert len(task_name) > 0, "At least one task must be provided"
        self.task_name = task_name

        if task_weight is None:
            self.task_weight = [1.0] * len(task_name)
        else:
            assert len(task_name) == len(
                task_weight
            ), "task_name and task_weight must have the same length"
            self.task_weight = task_weight

    def forward(self, output, target, data_id=None):
        losses = {}

        for task in self.task_name:
            if task == "gaze" and task in target:
                if self.compute_only_2d and data_id in [4, 9, 10]:
                    # apply only to 2d data
                    target_v = F.normalize(target[task][:, :, :2], p=2, dim=2, eps=1e-8)
                    output_v = F.normalize(output[task][:, :, :2], p=2, dim=2, eps=1e-8)
                else:
                    target_v = F.normalize(target[task], p=2, dim=2, eps=1e-8)
                    output_v = F.normalize(output[task], p=2, dim=2, eps=1e-8)

                mask = target["gaze_valid"]
                mask = rearrange(mask, "b t -> (b t)")

            if target_v is not None:
                # reshape the tensor to apply cosine similarity
                target_v = rearrange(target_v, "b t d -> (b t) d")
                output_v = rearrange(output_v, "b t d -> (b t) d")
                # apply mask with valid target
                target_v = target_v[mask]
                output_v = output_v[mask]
                sim = F.cosine_similarity(output_v, target_v, dim=1, eps=1e-8)
                sim = F.hardtanh_(sim, min_val=-1 + 1e-8, max_val=1 - 1e-8)
                loss = torch.acos(sim).mean() * 180 / math.pi
                losses[task] = loss
            else:
                losses[task] = 0

        loss = {t: w * losses[t] for t, w in zip(self.task_name, self.task_weight)}

        return loss
