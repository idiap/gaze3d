# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import io
import os
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.utils import load_pickle

DATASET_ID = {
    1: "gaze360",
    2: "gfie",
    3: "mpsgaze",
    4: "gazefollow",
    5: "childplay",
    6: "eyediap",
    7: "gaze360video",
    8: "gfievideo",
    9: "vat",
    10: "vatvideo",
    11: "eyediapvideo",
    12: "mpiiface",
    13: "childplayvideo",
}

# DATASET_ROOT = None

# DATASET_LOCATION = {
#     "gaze360image": {
#         "image_db": os.path.join(DATASET_ROOT,'Gaze360','gaze360_image_database.pkl'),
#         "sample_db": os.path.join(DATASET_ROOT,'Gaze360','samples/image_samples.csv'),
#         "clip_db": os.path.join(DATASET_ROOT,'Gaze360','gaze360_clip_database.pkl'),
#     },
#     "gaze360video":{
#         "image_db": os.path.join(DATASET_ROOT,'Gaze360','gaze360_image_database.pkl'),
#         "sample_db": os.path.join(DATASET_ROOT,'Gaze360','samples/image_samples.csv'),
#         "clip_db": os.path.join(DATASET_ROOT,'Gaze360','gaze360_clip_database.pkl'),
#     },
#     "gfieimage": {
#         "image_db": os.path.join(DATASET_ROOT,'GFIE','gfie_image_database.pkl'),
#         "sample_db": os.path.join(DATASET_ROOT,'GFIE','sample/samples.csv'),
#         "clip_db": os.path.join(DATASET_ROOT,'GFIE','gfie_clip_database.pkl'),
#     },
#     "gfievideo": {
#         "image_db": os.path.join(DATASET_ROOT,'GFIE','gfie_image_database.pkl'),
#         "sample_db": os.path.join(DATASET_ROOT,'GFIE','sample/samples.csv'),
#         "clip_db": os.path.join(DATASET_ROOT,'GFIE','gfie_clip_database.pkl'),
#     },
#     "eyediapimage" : {
#         "image_db": os.path.join(DATASET_ROOT,'Eyediap','eyediap_image_database.pkl'),
#         "sample_db": os.path.join(DATASET_ROOT,'Eyediap','eyediap_file.csv'),
#         "clip_db": os.path.join(DATASET_ROOT,'Eyediap','eyediap_clip_database.pkl'),
#     },
#     "eyediapvideo": {
#         "image_db": os.path.join(DATASET_ROOT,'Eyediap','eyediap_image_database.pkl'),
#         "sample_db": os.path.join(DATASET_ROOT,'Eyediap','eyediap_file.csv'),
#         "clip_db": os.path.join(DATASET_ROOT,'Eyediap','eyediap_clip_database.pkl'),
#     },
#     "mpiifaceimage": {
#         "image_db": os.path.join(DATASET_ROOT,'MPIIFace','mpiiface_image_database.pkl'),
#         "sample_db": os.path.join(DATASET_ROOT,'MPIIFace','mpiiface_file.csv'),
#         "clip_db": None,
#     },
#     "gazefollow": {
#         "image_db": os.path.join(DATASET_ROOT,'Gazefollow','gazefollow_image_database.pkl'),
#         "sample_db": os.path.join(DATASET_ROOT,'Gazefollow','gazefollow_file.csv'),
#         "clip_db": None,
#     },
# }

DATASET_LOCATION = {}


class DatabaseHandler:
    """
    Handle the different database format
    """

    @staticmethod
    def get_dataset_id(dataset_name: str):
        """
        Get the dataset id from the dataset name
        Args:
            dataset_name (str): The name of the dataset
        Returns:
            int: The id of the dataset
        """

        if dataset_name in DATASET_ID.values():
            return list(DATASET_ID.keys())[list(DATASET_ID.values()).index(dataset_name)]
        else:
            raise ValueError(f"Dataset {dataset_name} not found in {DATASET_ID}")

    @staticmethod
    def get_image_key_from_metadata(data_id: int, clip_id: int, frame_id: int, person_id: int):
        if data_id in [1, 2, 6, 7, 8, 11, 14, 15]:
            return f"clip_{clip_id:08d}_frame_{frame_id:08d}"
        elif data_id in [3, 4, 12]:
            return f"frame_{frame_id:08d}_face_{person_id:08d}"
        elif data_id in [5, 9, 10, 13]:
            return f"clip_{clip_id:08d}_frame_{frame_id:08d}_face_{person_id:08d}"
        else:
            raise ValueError(f"data_id not found {data_id}")

    @staticmethod
    def load_image_database(dataset_name: str):
        """Load the image database for the dataset"""
        path_image_db = DATASET_LOCATION[dataset_name]["image_db"]
        if not os.path.exists(path_image_db):
            raise ValueError(f"Image database {dataset_name} not found at {path_image_db}")
        return load_pickle(path_image_db)

    @staticmethod
    def load_sample_database(dataset_name: str):
        """Load the image database for the dataset"""
        path_sample_db = DATASET_LOCATION[dataset_name]["sample_db"]
        if not os.path.exists(path_sample_db):
            raise ValueError(f"Image database {dataset_name} not found at {path_sample_db}")
        return pd.read_csv(path_sample_db)

    @staticmethod
    def load_clip_database(dataset_name: str):
        """Load the clip database for the dataset"""
        path_clip_db = os.path.join(
            DATASET_LOCATION[dataset_name], f"{dataset_name}_clip_database.pkl"
        )
        if not os.path.exists(path_clip_db):
            raise ValueError(f"Clip database {dataset_name} not found at {path_clip_db}")
        return load_pickle(path_clip_db)

    @staticmethod
    def load_track_database(dataset_name: str):
        """Load the track database for the dataset"""
        path_track_db = os.path.join(
            DATASET_LOCATION[dataset_name], f"{dataset_name}_track_database.pkl"
        )
        if not os.path.exists(path_track_db):
            raise ValueError(f"Track database {dataset_name} not found at {path_track_db}")
        return load_pickle(path_track_db)


def get_bbox_in_body(bbox, body_bbox):
    bbox_in_body = np.zeros_like(bbox)
    bbox_in_body[0] = bbox[0] - body_bbox[0]
    bbox_in_body[1] = bbox[1] - body_bbox[1]
    bbox_in_body[2] = bbox[2] - body_bbox[0]
    bbox_in_body[3] = bbox[3] - body_bbox[1]
    bbox_in_body = bbox_in_body.astype(int)
    return bbox_in_body


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


def get_info_from_data(data, keys, info_keys, invalide_element=None):
    valide = []
    items = []
    for key in keys:
        if key not in data:
            valide.append(0)
            items.append(invalide_element)
        else:
            item = data[key]
            for info_key in info_keys:
                item = item[info_key]

            if np.equal(item, invalide_element).all():
                valide.append(0)
                items.append(invalide_element)
            else:
                valide.append(1)
                items.append(item)

    return valide, items


class BaseAnnotation:
    """
    used to store additional dataset information
    """

    def __init__(
        self,
        name,
        location,
    ):
        self.name = name
        self.location = location
        self.data = None

        self.load_data()

    def load_data(self):
        with open(self.location, "rb") as f:
            self.data = pickle.load(f)

    def get_data(self, key):
        return self.data[key]


class GazeAnnotation(BaseAnnotation):
    def __init__(self, location: str):
        super().__init__("gaze", location)

    def get_data(self, key):
        return self.data[key]["gaze_vector_pred"]


class BaseImage(Dataset):
    def __init__(
        self,
        data_name,
        image_db_path,
        sample_path,
        split,
        head_bbox_name,
        transform=None,
        additonal_data=[],
    ):
        self.data_name = data_name
        self.data_location = None
        self.image_db_path = image_db_path
        self.sample_path = sample_path
        self.split = split
        self.head_bbox_name = head_bbox_name
        self.read_mode = "pillow"
        self.transform = transform
        self.additonal_data = {add_data.name: add_data for add_data in additonal_data}
        self.base_data_dir = None
        self.base_slurm_dir = "/tmp"
        self.setup()

    def setup(self):
        # load image dataset
        with open(self.image_db_path, "rb") as f:
            self.data = pickle.load(f)
        # load the sample
        if self.sample_path is None:
            self.sample = pd.DataFrame(self.data.keys(), columns=["image_id"])
        else:
            self.sample = pd.read_csv(self.sample_path)
            if self.split != "all":
                self.sample = self.sample[self.sample["split"] == self.split]
            self.sample.reset_index(drop=True, inplace=True)

        # slurm data location
        if os.path.exists(os.path.join(self.base_slurm_dir, self.data_name)):
            self.data_location = True
            # print(f"using slurm data location for {self.data_name} dataset")
        else:
            self.data_location = False

    def check_data_is_defined(self):
        key_image = set(self.sample["image_id"].to_list())
        for k, v in self.additonal_data.items():
            key_data = set(v.data.keys())
            if len(key_image - key_data) != 0:
                print(f"missing data in {k} additional data")

    def __len__(self):
        return len(self.sample)

    def set_base_data_dir(self, base_data_dir):
        self.base_data_dir = base_data_dir

    def get_path_data(self, path):
        if self.data_location:
            new_path = path.replace(self.base_data_dir, self.base_slurm_dir)

            return new_path
        return path

    def read_image(self, image, frame=None, video_decoder=None):
        if self.read_mode == "pillow":
            if not isinstance(image, (str)):
                image = io.BytesIO(image)
            image = np.array(Image.open(image).convert("RGB"), copy=True)
        elif self.read_mode == "video":
            if frame is None and video_decoder is None:
                raise ValueError("frame must be set when using video decoder")
            image = video_decoder[frame - 1].asnumpy()
        else:
            raise NotImplementedError("use pillow or cv2")

        return image

    def __getitem__(self, idx):
        frames_info = self.data[self.sample.iloc[idx]["image_id"]]

        # input setup
        sample = {}
        sample["images"] = [self.read_image(self.get_path_data(frames_info["image_path_crop"]))]

        sample["frame_id"] = frames_info["frame"]
        sample["clip_id"] = int(frames_info["clip_id"].replace("clip_", ""))
        sample["head_bbox"] = [
            get_bbox_in_body(
                frames_info["other"][self.head_bbox_name], frames_info["other"]["head_bbox_crop"]
            )
        ]
        sample["bbox_strategy"] = "fixed_center"

        # task gaze
        gazes = [frames_info["other"]["gaze_dir"]]  # need to process the gaze cf gaze360 git
        gaze_float = torch.Tensor(np.array(gazes))
        normalized_gazes = torch.nn.functional.normalize(gaze_float)

        spherical_vector = torch.stack(
            (
                torch.atan2(normalized_gazes[:, 0], -normalized_gazes[:, 2]),
                torch.asin(normalized_gazes[:, 1]),
            ),
            1,
        )
        # only the gaze of the middle frame
        sample["task_gaze"] = spherical_vector[0]  # yaw pitch

        if self.transform:
            sample = self.transform(sample)

        return sample
