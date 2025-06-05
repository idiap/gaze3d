# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Iterator, List

import torch
from PIL import Image
from torch.utils.data import Sampler

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

"""
* BatchSamplerCombined: Class based on BatchSampler in pytorch
* BatchSamplerSequential: Class based on BatchSampler in pytorch
* default_loader: Load image using PIL
"""


def default_loader(path):
    try:
        im = Image.open(path).convert("RGB")
        return im
    except OSError:
        raise OSError(f"Cannot load image {path}")
        print(path)
        return Image.new("RGB", (224, 224), "white")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# class based on BatchSampler in pytorch
class BatchSamplerCombined(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.
    each batch sample one batch from one dataset according to a sample probability.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self,
        samplers: List[Sampler[int]],
        data_size: List[int],
        batch_size: int,
        generator=None,
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )

        self.samplers = samplers
        self.data_size = data_size
        self.batch_size = batch_size
        self.drop_last = True
        self.num_sample_per_dataset = [
            len(sampler) // self.batch_size for sampler in self.samplers
        ]
        self.num_samples = sum([len(sampler) // self.batch_size for sampler in self.samplers])
        self.generator = generator

    def get_data_index(self, i):
        if i == 0:
            return 0
        return sum(self.data_size[:i])

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951

        # Randomly shuffle the batch selection
        data_selection = torch.concatenate(
            [
                torch.full((self.num_sample_per_dataset[i],), i, dtype=torch.int64)
                for i in range(len(self.samplers))
            ]
        )
        data_selection = data_selection[
            torch.randperm(len(data_selection), generator=self.generator)
        ]
        samplers_iter = [iter(sampler) for sampler in self.samplers]

        # Add the data index to account for the offset index after concat datasets
        for i in data_selection:
            batch = [
                next(samplers_iter[i]) + self.get_data_index(i) for _ in range(self.batch_size)
            ]
            yield batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]

        return self.num_samples


class BatchSamplerSequential(Sampler[List[int]]):
    r"""Wraps sequential samplers in batch for mulitple datasets.
    It is mainly used for validataion and test dataloader inorder to have sequential batch without drop last
    Warning: batch at the end of the dataset can be smaller than batch_size
    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(SequentialBatchSampler(SequentialSampler(range(10)), batch_size=3))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """

    def __init__(self, samplers: List[Sampler[int]], batch_size) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.

        self.batch_sampler = [
            torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
            for sampler in samplers
        ]
        self.data_size = [len(sampler) for sampler in samplers]

    def get_data_index(self, i):
        if i == 0:
            return 0
        return sum(self.data_size[:i])

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951

        for idx_data, batch_sampler in enumerate(self.batch_sampler):
            for batch in batch_sampler:
                yield [batch[i] + self.get_data_index(idx_data) for i in range(len(batch))]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]

        return sum([len(batch_sampler) for batch_sampler in self.batch_sampler])
