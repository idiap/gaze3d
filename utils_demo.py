# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import io
import pickle

import av
import cv2
import numpy as np
import pandas as pd
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from src.data.components.transforms import (
    BboxReshape,
    Concatenate,
    Crop,
    Normalize,
    ToImage,
    ToTensor,
)

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

TRANSFORM = Compose(
    [
        BboxReshape(
            square=True,
            ratio=-0.1,
        ),
        ToImage(),
        Crop(
            output_size=224,
        ),
        Concatenate(),
        ToTensor(),
        Normalize(
            mean=IMG_MEAN,
            std=IMG_STD,
        ),
    ]
)


def identify_modality(file_path):
    if file_path.endswith(".jpg") or file_path.endswith(".png"):
        return "image"
    elif file_path.endswith(".mp4") or file_path.endswith(".avi"):
        return "video"
    else:
        raise ValueError("File extension not supported {}".format(file_path))


def read_image(image, frame=None, video_decoder=None, read_mode="pillow"):
    if read_mode == "pillow":
        if not isinstance(image, (str)):
            image = io.BytesIO(image)
        image = np.array(Image.open(image).convert("RGB"), copy=True)
    elif read_mode == "video":
        if frame is None and video_decoder is None:
            raise ValueError("frame must be set when using video decoder")
        try:
            image = video_decoder[int(frame - 1)].asnumpy().astype(np.uint8)
        except:
            image = None

    else:
        raise NotImplementedError("use pillow or cv2")

    return image


def extract_frame(video_path, frame_number):
    """
    Extracts a specific frame from a video using PyAV.

    Args:
        video_path (str): Path to the video file.
        frame_number (int): The frame number to extract.

    Returns:
        torch.Tensor: The extracted frame as a tensor in CHW format.
    """
    with av.open(video_path) as container:
        video_stream = container.streams.video[0]
        fps = video_stream.average_rate  # Frames per second
        time_base = video_stream.time_base  # Time base of the stream

        # Calculate the timestamp for the desired frame
        target_pts = int(frame_number / fps / time_base)

        # Seek to the nearest keyframe before the target timestamp
        container.seek(target_pts, stream=video_stream)

        for frame in container.decode(video_stream):
            if frame.pts >= target_pts:
                return frame.to_image()

    raise ValueError(f"Frame {frame_number} not found in {video_path}")


def create_window(frame: int, window_size: int, window_stride: int):
    """Create a window of frames around the current frame

    Args:
        frame (int): The current frame
        window_size (int): The size of the window
        window_stride (int): The stride of the window

    Returns:
        List[int]: A list of frame indices
    """
    assert window_size % 2 == 1, "Window size must be odd"

    window_min = frame - (window_size // 2) * window_stride
    window_max = frame + ((window_size // 2) + 1) * window_stride
    return np.arange(window_min, window_max + 1, window_stride)


class DemoImageData(Dataset):
    def __init__(self, detected_head_file, input_file_path):
        if isinstance(detected_head_file, str):
            self.detected_head = pd.read_csv(detected_head_file)
        else:
            self.detected_head = detected_head_file
        self.input_file_path = input_file_path
        self.transforms = TRANSFORM
        # handle input modality
        self.input_modality = identify_modality(input_file_path)

    def __len__(self):
        return len(self.detected_head)

    def __getitem__(self, idx):
        if self.input_modality == "video":
            read_mode = "video"
            with open(self.input_file_path, "rb") as f:
                video_decoder = VideoReader(self.input_file_path)
        else:
            read_mode = "pillow"
            video_decoder = None

        frame_info = self.detected_head.iloc[idx]

        sample = {}
        sample["images"] = [
            read_image(
                self.input_file_path,
                frame=frame_info["frame_id"],
                video_decoder=video_decoder,
                read_mode=read_mode,
            )
        ]
        sample["head_bbox"] = [
            [
                frame_info["xmin"],
                frame_info["ymin"],
                frame_info["xmax"],
                frame_info["ymax"],
            ]
        ]
        sample = self.transforms(sample)

        return sample


class DemoVideoData(Dataset):
    def __init__(self, detected_head_file, input_file_path, window_stride=1):
        self.input_modality = identify_modality(input_file_path)
        assert self.input_modality == "video"
        self.read_mode = "video"
        cap = cv2.VideoCapture(input_file_path)
        self.fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

        self.detected_head = pd.read_csv(detected_head_file)
        self.valid_frame = set(self.detected_head["frame_id"])
        self.input_file_path = input_file_path
        self.transforms = TRANSFORM

        self.window_size = 7
        # self.window_stride = int(self.fps) // 2
        self.window_stride = window_stride

    def __len__(self):
        return len(self.detected_head)

    def __getitem__(self, idx):
        # with open(self.input_file_path, 'rb') as f:
        #     self.video_decoder = VideoReader(self.input_file_path)

        # load the image
        frame_info = self.detected_head.iloc[idx]

        # filter person track
        tracks = self.detected_head[self.detected_head["pid"] == frame_info["pid"]]
        tracks = tracks.reset_index(drop=True)
        tracks_frame = set(tracks["frame_id"])
        tracks_frame_head = {
            frame: [xmin, ymin, xmax, ymax]
            for frame, xmin, ymin, xmax, ymax in zip(
                tracks["frame_id"], tracks["xmin"], tracks["ymin"], tracks["xmax"], tracks["ymax"]
            )
        }

        # image = read_image(self.input_file_path,
        #                    frame=frame_info["frame_id"],
        #                    video_decoder=self.video_decoder,
        #                    read_mode=self.read_mode)

        image = extract_frame(self.input_file_path, int(frame_info["frame_id"] - 1))

        sequence_frame = create_window(
            frame_info["frame_id"], self.window_size, self.window_stride
        )
        valid_frame = [frame in tracks_frame for frame in sequence_frame]

        sample = {}
        sample["images"] = [
            extract_frame(self.input_file_path, int(frame - 1))
            if valid
            else np.zeros_like(image).astype(np.uint8)
            for frame, valid in zip(sequence_frame, valid_frame)
        ]
        sample["head_bbox"] = [
            tracks_frame_head[frame]
            if frame in tracks_frame_head
            else tracks_frame_head[frame_info["frame_id"]]
            for frame in sequence_frame
        ]
        sample["bbox_strategy"] = "followed"

        sample = self.transforms(sample)

        return sample
