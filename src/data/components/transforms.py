# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import hydra
import numpy as np
import torch
import torchvision.transforms.v2 as tf
from omegaconf import DictConfig
from torchvision.ops import box_convert
from torchvision.transforms import Compose


class TransformsWrapper:
    def __init__(self, transforms_cfg: DictConfig) -> None:
        """TransformsWrapper module.
        Args:
            transforms_cfg (DictConfig): Transforms config.
        """
        augmentations = []
        if not transforms_cfg.get("order"):
            raise RuntimeError(
                "TransformsWrapper requires param <order>, i.e."
                "order of augmentations as List[augmentation name]"
            )
        for augmentation_name in transforms_cfg.get("order"):
            augmentation = hydra.utils.instantiate(transforms_cfg.get(augmentation_name))
            if augmentation is not None:
                augmentations.append(augmentation)
        self.augmentations = Compose(augmentations)

    def get_transforms(self):
        """Get TransformsWrapper module.
        Returns:
            Any: Transformation results.
        """
        return self.augmentations


def reshape_bbox_adjust_top(bbox, ratio=0.0, square=True):
    """
    Reshape the head bounding box to include more of the
    body and head.
    """
    bbox_x_middle = (bbox[:, 0:1] + bbox[:, 2:3]) / 2
    bbox_xymin = bbox[:, :2]
    bbox_xywh = box_convert(bbox, in_fmt="xyxy", out_fmt="xywh")

    if square:
        sizes = torch.max(bbox_xywh[:, 2:], 1)[0]
        sizes = torch.stack([sizes, sizes], 1)
    else:
        sizes = bbox_xywh[:, 2:]

    sizes = sizes * (1 + ratio)
    bbox_xmin = bbox_x_middle - (sizes[:, 0:1] / 2)

    if ratio > 0.0:
        # 7% of the height to make sure the top of the head is included
        bbox_ymin = bbox_xymin[:, 1:2] - ((sizes[:, 1:2] * ratio) * 0.07)
    else:
        bbox_ymin = bbox_xymin[:, 1:2] - ((sizes[:, 1:2] * ratio))

    bbox_new = torch.cat([bbox_xmin, bbox_ymin, sizes], 1)
    return box_convert(bbox_new, in_fmt="xywh", out_fmt="xyxy")


def reshape_bbox_adjust_center(bbox, ratio=0.0, square=True):
    bbox_cxcywh = box_convert(bbox, in_fmt="xyxy", out_fmt="cxcywh")

    if square:
        sizes = torch.max(bbox_cxcywh[:, 2:], 1)[0]
        sizes = torch.stack([sizes, sizes], 1)
    else:
        sizes = bbox_cxcywh[:, 2:]

    bbox_cxcywh[:, 2:] = sizes + (sizes * ratio)
    return box_convert(bbox_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")


class BboxReshape(object):
    """
    Reshape the bounding box of the head from yolo.
    We want to make sure to extend the bbox such that it contains as much as possible
    of the body and head.
    """

    def __init__(self, square: bool = True, ratio: float = 0.0, adjust_top=True):
        self.square = square
        self.ratio = ratio
        self.adjust_top = adjust_top

    def __call__(self, sample):
        bbox = sample["head_bbox"]

        if not isinstance(bbox, torch.Tensor):
            bbox = torch.tensor(np.array(bbox))

        if self.adjust_top:
            sample["head_bbox"] = reshape_bbox_adjust_top(
                bbox, ratio=self.ratio, square=self.square
            )
        else:
            sample["head_bbox"] = reshape_bbox_adjust_center(
                bbox, ratio=self.ratio, square=self.square
            )

        return sample


class Crop(object):
    """Crop according to the bbox in the sample

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        head_bbox = sample["head_bbox"]
        head_bbox = box_convert(head_bbox, in_fmt="xyxy", out_fmt="xywh")
        head_bbox = head_bbox.type(torch.int16)
        # crop and resize functions
        sample["images"] = [
            tf.functional.resized_crop(
                img,
                head_bbox[i, 1],
                head_bbox[i, 0],
                head_bbox[i, 3],
                head_bbox[i, 2],
                self.output_size,
                antialias=True,
            )
            for i, img in enumerate(sample["images"])
        ]

        return sample


class HorizontalFlip(object):
    """Flip the images in the sample horizontally"""

    def __init__(self, p=0.5, verbose=False):
        self.p = p
        self.verbose = verbose

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            if self.verbose:
                print("Applying horizontal flip")

            sample["images"] = tf.functional.hflip(sample["images"])

            if "gaze" in sample:
                if sample["gaze"].size(1) == 2:
                    sample["gaze"] = sample["gaze"] * torch.tensor([-1, 1])  # x is flipped
                else:
                    sample["gaze"] = sample["gaze"] * torch.tensor([-1, 1, 1])  # x is flipped

        return sample


class ColorJitter(object):
    """
    Applies random colors transformations to the input (ie. brightness,
    contrast, saturation and hue).
    """

    def __init__(
        self,
        brightness=(0.5, 1.5),
        contrast=(0.5, 1.5),
        saturation=(0.0, 1.5),
        hue=None,
        p=0.5,
        verbose=False,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
        self.verbose = verbose

    def __call__(self, sample):
        if torch.rand(1) <= self.p:
            if self.verbose:
                print("Applying color jittering")
            # Sample color transformation factors and order
            brightness_factor = (
                None if self.brightness is None else torch.rand(1).uniform_(*self.brightness)
            )
            contrast_factor = (
                None if self.contrast is None else torch.rand(1).uniform_(*self.contrast)
            )
            saturation_factor = (
                None if self.saturation is None else torch.rand(1).uniform_(*self.saturation)
            )
            hue_factor = None if self.hue is None else torch.rand(1).uniform_(*self.hue)

            if np.array(sample["images"]).sum() > 0:
                fn_indices = torch.randperm(4)
                for fn_id in fn_indices:
                    if fn_id == 0 and brightness_factor is not None:
                        sample["images"] = tf.functional.adjust_brightness(
                            sample["images"], brightness_factor
                        )

                    elif fn_id == 1 and contrast_factor is not None:
                        sample["images"] = tf.functional.adjust_contrast(
                            sample["images"], contrast_factor
                        )

                    elif fn_id == 2 and saturation_factor is not None:
                        sample["images"] = tf.functional.adjust_saturation(
                            sample["images"], saturation_factor
                        )

                    elif fn_id == 3 and hue_factor is not None:
                        sample["images"] = tf.functional.adjust_hue(sample["images"], hue_factor)

        return sample


class RandomGaussianBlur(object):
    """Apply random gaussian blur to the input"""

    def __init__(self, radius=4, sigma=(0.2, 2.0), p=0.5, verbose=False):
        self.p = p
        self.radius = radius
        self.sigma = sigma
        self.verbose = verbose

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            if self.verbose:
                print("Applying random gaussian blur")
            kernel_size = torch.randint(1, self.radius, (1,)).item() * 2 + 1
            sample["images"] = tf.functional.gaussian_blur(
                sample["images"], kernel_size, self.sigma
            )

        return sample


class ToImage(object):
    """Convert PIL image to Tensor.
    ref: https://pytorch.org/vision/main/transforms.html#range-and-dtype"""

    def __call__(self, sample):
        sample["images"] = [
            tf.functional.to_dtype(tf.functional.to_image(img), dtype=torch.uint8, scale=True)
            for img in sample["images"]
        ]
        return sample


class Concatenate(object):
    """Concatenate the images in the sample"""

    def __call__(self, sample):
        # stack the images along the first dimension
        sample["images"] = torch.stack(sample["images"], 0)
        return sample


class ToTensor(object):
    """Convert tensor image to float"""

    def __call__(self, sample):
        sample["images"] = tf.functional.to_dtype(
            sample["images"], dtype=torch.float32, scale=True
        )
        return sample


class Normalize(object):
    """Normalize the images in the sample"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        # B, C, H, W
        sample["images"] = tf.functional.normalize(sample["images"], self.mean, self.std)
        return sample
