# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import pickle
import warnings
from typing import Any, Dict, List, Union

import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning
from torchmetrics import Metric
from tqdm import tqdm

from src.data.components.base_dataset import DATASET_ID, DatabaseHandler
from src.utils.metrics_utils import compute_angular_error_cartesian, spherical2cartesial

# filter the warnings about ill-defined P,R and F1
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)


class AngularError(Metric):
    """Angular error metric for gaze estimation used in Gaze360"""

    def __init__(self, only_middle=True):
        super().__init__()
        self.only_middle = only_middle
        self.add_state("total_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, input, target):
        self.total_error += compute_angular_error_cartesian(input, target, self.only_middle)
        self.total_samples += input.size(0)

    def compute(self):
        return self.total_error / self.total_samples


class PredictionSave(Metric):
    def __init__(self) -> None:
        super().__init__(compute_on_cpu=True, compute_with_cache=False)

        self.add_state("frame_pred", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("frame_gt", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("frame_id", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("video_id", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("person_id", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("data_id", default=[], dist_reduce_fx="cat", persistent=True)

    def update(self, pred, gt, frame_id, video_id, person_id, data_id):
        self.frame_pred += pred
        self.frame_gt += gt
        self.frame_id += frame_id
        self.video_id += video_id
        self.person_id += person_id
        self.data_id += data_id

    def compute(self):
        if len(self.frame_pred) == 0:
            return {
                "frame_pred": None,
                "frame_gt": None,
                "frame_id": None,
                "video_id": None,
                "person_id": None,
                "data_id": None,
            }

        dict_out = {}
        nb_pred = len(self.frame_pred)
        print(f"nb_pred: {nb_pred}")
        print(f"frame_id: {len(self.frame_id)}")
        print(f"video_id: {len(self.video_id)}")
        print(f"person_id: {len(self.person_id)}")
        print(f"data_id: {len(self.data_id)}")

        for i in tqdm(range(nb_pred)):
            data_id = torch.unique(self.data_id[i]).numpy()
            assert len(data_id) == 1, "Data_id should be unique by batch"
            data_id = data_id[0]

            if DATASET_ID[data_id] not in dict_out.keys():
                dict_out[DATASET_ID[data_id]] = {
                    "frame_pred": [self.frame_pred[i]],
                    "frame_gt": [self.frame_gt[i]],
                    "frame_id": [self.frame_id[i]],
                    "video_id": [self.video_id[i]],
                    "person_id": [self.person_id[i]],
                }
            else:
                dict_out[DATASET_ID[data_id]]["frame_pred"].append(self.frame_pred[i])
                dict_out[DATASET_ID[data_id]]["frame_gt"].append(self.frame_gt[i])
                dict_out[DATASET_ID[data_id]]["frame_id"].append(self.frame_id[i])
                dict_out[DATASET_ID[data_id]]["video_id"].append(self.video_id[i])
                dict_out[DATASET_ID[data_id]]["person_id"].append(self.person_id[i])

        for data_name, data in dict_out.items():
            dict_out[data_name]["frame_pred"] = torch.stack(data["frame_pred"]).cpu()
            dict_out[data_name]["frame_gt"] = torch.stack(data["frame_gt"]).cpu()
            dict_out[data_name]["frame_id"] = torch.stack(data["frame_id"]).cpu()
            dict_out[data_name]["video_id"] = torch.stack(data["video_id"]).cpu()
            dict_out[data_name]["person_id"] = torch.stack(data["person_id"]).cpu()

        return dict_out

    def save(self, output_file):
        gather_prediction = self.compute()
        with open(output_file, "wb") as f:
            pickle.dump(gather_prediction, f)

        return gather_prediction


class DatasetManager:
    def __init__(self, dataset_name, dataset_name_short, data=None):
        self.data_handler = DatabaseHandler()
        self.dataset_name = dataset_name
        self.dataset_name_short = dataset_name_short
        self.dataset_id = self.data_handler.get_dataset_id(dataset_name)
        if data:
            self.image_db = data.image_db
            self.sample_db = data.sample_db
        else:
            self.image_db = self.data_handler.load_image_database(self.dataset_name)
            self.sample_db = self.data_handler.load_sample_database(self.dataset_name)

        self.image_key_test_subset = None

    def prepare_dataset(self):
        raise NotImplementedError("prepare_dataset method should be implemented in the subclass")

    def create_key_test_subset(self):
        """
        need to create :
        - image_key_test_subset: a dict with the name of the subset and a list a image key to compute the metrics on
        """
        raise NotImplementedError(
            "create_test_subset method should be implemented in the subclass"
        )

    def process(self):
        """
        This function is called to process the dataset.
        It will prepare the dataset and create the test subset.
        """
        self.prepare_dataset()
        self.create_key_test_subset()


class Gaze360Image(DatasetManager):
    def __init__(self, dataset_name="gaze360image", dataset_name_short="G360I", data=None):
        super().__init__(dataset_name, dataset_name_short, data=data)
        self.process()

    def prepare_dataset(self):
        # include the face information in the prediction
        for k in self.image_db.keys():
            face_info = self.image_db[k]["other"]["person_face_bbox"]
            is_face = face_info[0] != -1
            self.image_db[k]["face_info"] = 1 if is_face else 0
            gaze_dir = self.image_db[k]["other"]["gaze_dir"]
            # compute the angular error with center (0,0,-1)
            angular_error = (
                180
                / np.pi
                * np.arccos(
                    np.dot(gaze_dir, np.array([0, 0, -1]))
                    / (np.linalg.norm(gaze_dir) * np.linalg.norm(np.array([0, 0, -1])))
                )
            )
            self.image_db[k]["angular_error"] = angular_error

    def create_key_test_subset(self):
        self.image_key_test_subset = {}

        subset_full = self.sample_db[self.sample_db["split"] == "test"]["image_id"].tolist()
        subset_back = [k for k in subset_full if self.image_db[k]["angular_error"] > 90]
        subset_180 = [k for k in subset_full if self.image_db[k]["angular_error"] <= 90]
        subset_40 = [k for k in subset_full if self.image_db[k]["angular_error"] <= 20]
        subset_face = [k for k in subset_full if self.image_db[k]["face_info"] == 1]
        subset_face_180 = [
            k
            for k in subset_180
            if self.image_db[k]["face_info"] == 1 and self.image_db[k]["angular_error"] <= 90
        ]
        subset_face_40 = [
            k
            for k in subset_40
            if self.image_db[k]["face_info"] == 1 and self.image_db[k]["angular_error"] <= 20
        ]

        self.image_key_test_subset = {
            "Full": set(subset_full),
            "Back": set(subset_back),
            "180": set(subset_180),
            "40": set(subset_40),
            "Face": set(subset_face),
            "Face_180": set(subset_face_180),
            "Face_40": set(subset_face_40),
        }


class Gaze360Video(Gaze360Image):
    def __init__(self, dataset_name="gaze360video", dataset_name_short="G360V", data=None):
        super().__init__(dataset_name, dataset_name_short, data=data)


class GFIEImage(DatasetManager):
    def __init__(self, dataset_name="gfieimage", dataset_name_short="GFIEI"):
        super().__init__(dataset_name, dataset_name_short)
        self.process()

    def prepare_dataset(self):
        # include the face information in the prediction
        image_keys_test = self.sample_db[self.sample_db["split"] == "test"]["image_id"].tolist()
        for k in image_keys_test:
            gaze_dir = self.image_db[k]["other"]["gaze_direction"]
            # compute the angular error with center (0,0,-1)
            angular_error = (
                180
                / np.pi
                * np.arccos(
                    np.dot(gaze_dir, np.array([0, 0, -1]))
                    / (np.linalg.norm(gaze_dir) * np.linalg.norm(np.array([0, 0, -1])))
                )
            )
            self.image_db[k]["angular_error"] = angular_error

    def create_key_test_subset(self):
        self.image_key_test_subset = {}

        subset_full = self.sample_db[self.sample_db["split"] == "test"]["image_id"].tolist()
        subset_back = [k for k in subset_full if self.image_db[k]["angular_error"] > 90]
        subset_180 = [k for k in subset_full if self.image_db[k]["angular_error"] <= 90]

        self.image_key_test_subset = {
            "Full": set(subset_full),
            "Back": set(subset_back),
            "180": set(subset_180),
        }


class GFIEVideo(GFIEImage):
    def __init__(self, dataset_name="gfievideo", dataset_name_short="GFIEV", data=None):
        super().__init__(dataset_name, dataset_name_short, data=data)


class EyediapImage(DatasetManager):
    def __init__(self, dataset_name="eyediapimage", dataset_name_short="EDIAPI", data=None):
        super().__init__(dataset_name, dataset_name_short, data=data)
        self.process()

    def prepare_dataset(self):
        pass

    def create_key_test_subset(self):
        self.image_key_test_subset = {}
        subset_test = self.sample_db[self.sample_db["split"] == "test"]["image_id"].tolist()
        subset_FT = [k for k in subset_test if self.image_db[k]["task"] == "FT"]
        subset_FT_M = [k for k in subset_FT if self.image_db[k]["static"] == "M"]
        subset_FT_S = [k for k in subset_FT if self.image_db[k]["static"] == "S"]
        subset_CS = [k for k in subset_test if self.image_db[k]["task"] == "CS"]

        self.image_key_test_subset = {
            "FT": set(subset_FT),
            "FT_M": set(subset_FT_M),
            "FT_S": set(subset_FT_S),
            "CS": set(subset_CS),
        }


class EyediapVideo(EyediapImage):
    def __init__(self, dataset_name="eyediapvideo", dataset_name_short="EDIAPV", data=None):
        super().__init__(dataset_name, dataset_name_short, data=data)


class MPIIFaceImage(DatasetManager):
    def __init__(self, dataset_name="mpiifaceimage", dataset_name_short="MPII", data=None):
        super().__init__(dataset_name, dataset_name_short, data=data)
        self.process()

    def prepare_dataset(self):
        pass

    def create_key_test_subset(self):
        self.image_key_test_subset = {}
        subset_test = self.sample_db[self.sample_db["split"] == "test"]["image_id"].tolist()
        self.image_key_test_subset = {
            "All": set(subset_test),
        }


class VATImage(DatasetManager):
    def __init__(self, dataset_name="vatimage", dataset_name_short="VATI", data=None):
        super().__init__(dataset_name, dataset_name_short, data=data)
        self.process()

    def prepare_dataset(self):
        pass

    def create_key_test_subset(self):
        self.image_key_test_subset = {}
        subset_test = self.sample_db[self.sample_db["split"] == "test"]["image_id"].tolist()
        self.image_key_test_subset = {
            "All": set(subset_test),
        }


class TaskMetric:
    def __init__(self, data: DatasetManager):
        self.data = data
        self.image_idx_test_subset = {}

        # prediction test is the file output by the model which is a dict with
        # - dict_keys(['frame_pred', 'frame_gt', 'frame_id', 'video_id', 'person_id'])
        # - the size is [batch_size, t, ...] for frame_pred and frame_gt
        self.prediction_test = None

    def compute(self):
        """
        return the metric results in a format datasetname/task/subset_name
        """
        self.metric_results = {}
        return self.metric_results

    def load_prediction(self, prediction: Dict):
        """
        It can handle different prediction format:
        1. {'frame_pred': ..., 'frame_gt': ...}
        2. { 'dataset_name': {'frame_pred': ..., 'frame_gt': ...} }
        """
        # check if exp_results is a dict
        if isinstance(prediction, str):
            with open(prediction, "rb") as f:
                self.prediction_test = pickle.load(f)
        elif isinstance(prediction, dict):
            self.prediction_test = prediction
        else:
            raise ValueError("prediction should be a dict or a path to a pickle file")

        if "frame_pred" not in self.prediction_test:
            dataset_in_prediction = [*self.prediction_test]
            filter_dataset = [k for k in dataset_in_prediction if self.data.dataset_name in k]
            if len(filter_dataset) == 0:
                print(f"dataset {self.data.dataset_name} not found in the prediction file")
                self.prediction_test = None
            else:
                assert (
                    len(filter_dataset) == 1
                ), f"multiple dataset found in the prediction file {filter_dataset}"
                self.prediction_test = self.prediction_test[filter_dataset[0]]

    def display_test_info(self):
        """
        Display the number of samples in each subset
        """
        print(
            f"Number of samples in the prediction set: {len(self.prediction_test['frame_pred'])}"
        )
        for subset_name, subset_keys in self.data.image_key_test_subset.items():
            print(f"Samples in the {subset_name} subset: {len(subset_keys)}")

    def create_idx_subset(self) -> Dict[str, List[int]]:
        self.image_idx_test_subset = {k: [] for k in self.data.image_key_test_subset}
        for idx in tqdm(
            range(len(self.prediction_test["frame_pred"])), desc="Creating test subset idx"
        ):
            image_key = self.data.data_handler.get_image_key_from_metadata(
                data_id=self.data.dataset_id,
                clip_id=self.prediction_test["video_id"][idx].int().item(),
                frame_id=self.prediction_test["frame_id"][idx].int().item(),
                person_id=self.prediction_test["person_id"][idx].int().item(),
            )
            for subset_name, subset_image_keys in self.data.image_key_test_subset.items():
                if image_key in subset_image_keys:
                    self.image_idx_test_subset[subset_name].append(idx)

        for subset_name, subset_idx in self.image_idx_test_subset.items():
            assert len(subset_idx) == len(
                self.data.image_key_test_subset[subset_name]
            ), f"the subset {subset_name} doesn't match the size, prediction are missing"

    def process(self, prediction, verbose=False):
        self.load_prediction(prediction)
        if self.prediction_test is None:
            return {}
        self.create_idx_subset()
        if verbose:
            self.display_test_info()
        metric_results = self.compute()
        return metric_results


class GazeMetric(TaskMetric):
    def __init__(self, data: DatasetManager):
        super().__init__(data)

    def compute(self):
        metric_subsets = {
            k: AngularError(only_middle=True, dataset="data") for k in self.image_idx_test_subset
        }

        for subset_name, subset_idx in self.image_idx_test_subset.items():
            pred = self.prediction_test["frame_pred"][subset_idx]
            target = self.prediction_test["frame_gt"][subset_idx]
            metric_subsets[subset_name].update(pred, target)
            # compute the angular error for each sample in the test set
            metric_subsets[subset_name] = metric_subsets[subset_name].compute()

        metric_results = {
            f"{self.data.dataset_name_short}/gaze/{k}": v["data/gaze/angular_error"].item()
            for k, v in metric_subsets.items()
        }
        return metric_results
