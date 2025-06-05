# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import os
import shlex
import subprocess as sp
import warnings
from functools import partial

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from boxmot import OCSORT
from matplotlib import colormaps
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.gat_model import GaT, HeadDict, MLPHead, Swin3D
from utils_demo import DemoImageData, DemoVideoData, identify_modality

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# ================================ ARGS ================================ #
parser = argparse.ArgumentParser(description="Predict gaze on videos")
parser.add_argument(
    "--input-filename", type=str, help="Name of the clip file to process (with extension)."
)
parser.add_argument(
    "--output-dir", type=str, default="data", help="Name of the folder where to save the output."
)

# Model
parser.add_argument(
    "--ckpt-path",
    type=str,
    default="./checkpoints/gat_stwsge_gaze360_gf.ckpt",
    help="Path to the pre-trained model checkpoint.",
)
parser.add_argument(
    "--modality",
    type=str,
    default="image",
    help="In case of video process image independently or using sliding window.",
)
parser.add_argument(
    "--window-stride", type=int, default=1, help="Stride used in the sliding window."
)
parser.add_argument("--device", type=str, default="cpu", help="Device to use for inference.")
parser.add_argument(
    "--batch-size", type=int, default=24, help="Batch size that can fit in RAM or GPU."
)
parser.add_argument(
    "--num-workers", type=int, default=4, help="Number of worker for multiprocessing dataloader."
)

args = parser.parse_args()


# =============================== GLOBALS =============================== #
CMAP = colormaps.get_cmap("brg")
COLOR_NAMES = [
    "mediumvioletred",
    "green",
    "dodgerblue",
    "crimson",
    "goldenrod",
    "DarkSlateGray",
    "saddlebrown",
    "purple",
    "teal",
]
COLORS = [
    (199, 21, 133),
    (0, 128, 0),
    (30, 144, 255),
    (220, 20, 60),
    (218, 165, 32),
    (47, 79, 79),
    (139, 69, 19),
    (128, 0, 128),
    (0, 128, 128),
]
DET_THR = 0.4  # head detection threshold

# ========================= UTILITY FUNCTIONS =========================== #


def load_tracker():
    tracker = OCSORT()
    return tracker


def load_head_detection_model(device):
    # Load and return the pre-trained head detection model
    ckpt_path = "./weights/crowdhuman_yolov5m.pt"
    model = torch.hub.load("ultralytics/yolov5", "custom", path=ckpt_path, verbose=False)
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.classes = [1]  # filter by class, i.e. = [1] for heads
    model.amp = False  # Automatic Mixed Precision (AMP) inference
    model = model.to(device)
    model.eval()
    return model


def detect_heads(image, model):
    """
    Detect heads in the image using the provided model.
    Returns a numpy array containing the detected head bboxes and their confidence scores.
    """
    detections = (
        model(image, size=640).pred[0].cpu().numpy()[:, :-1]
    )  # filter out the class column
    return detections


def load_gaze_model(ckpt_path, device):
    # Load and return the pre-trained Gaze-At-Target model
    # Load checkpoint
    model = GaT(
        encoder=Swin3D(pretrained=False),
        head_dict=HeadDict(
            names=["gaze"],
            modules=[
                partial(
                    MLPHead,
                    hidden_dim=256,
                    num_layers=1,
                    out_features=3,
                )
            ],
        ),
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def draw_arrow2D(
    image,
    gaze,
    position=None,
    head_size=None,
    d=0.1,
    color=(255, 0, 0),
    thickness=10,
):
    w, h = image.shape[1], image.shape[0]

    if position is None:
        position = [w // 2, h // 2]

    if head_size:
        length = head_size
    else:
        length = w * d

    gaze_dir = gaze / np.linalg.norm(gaze)
    dx = -length * gaze_dir[0]
    dy = -length * gaze_dir[1]

    cv2.arrowedLine(
        image,
        tuple(np.round(position).astype(np.int32)),
        tuple(np.round([position[0] + dx, position[1] + dy]).astype(int)),
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.2,
    )
    return image


def draw_gaze(
    image,
    head_bbox,
    head_pid,
    gaze,
    cmap,
    colors,
    thickness=10,
    thickness_gaze=10,
    fs=0.8,
):
    img_h, img_w = image.shape[0], image.shape[1]
    scale = max(img_h, img_w) / 1920
    fs *= scale
    thickness = int(scale * thickness)
    thickness_gaze = int(scale * thickness_gaze)

    # =============== Draw Prediction =============== #

    xmin, ymin, xmax, ymax = head_bbox[0], head_bbox[1], head_bbox[2], head_bbox[3]
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

    # Compute Head Center
    head_center = np.array([(xmin + xmax) // 2, (ymin + ymax) // 2])
    head_radius = max(xmax - xmin, ymax - ymin) // 2
    head_radius = int(head_radius * 1.2)  # enlarge the head circle
    color = colors[head_pid % len(colors)]
    cv2.circle(image, head_center, head_radius + 1, color, thickness)  # head circle

    # Draw header
    header_text = f"P{int(head_pid)}"
    (w_text, h_text), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    header_ul = (
        int(head_center[0] - w_text / 2),
        int(head_center[1] - head_radius - 1 - thickness / 2),
    )
    header_br = (
        int(head_center[0] + w_text / 2),
        int(head_center[1] - head_radius - 1 + h_text + 5),
    )
    cv2.rectangle(image, header_ul, header_br, color, -1)  # header bbox
    cv2.putText(
        image,
        header_text,
        (header_ul[0], int(head_center[1] - head_radius - 1 + h_text)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )  # header text

    # Draw 3D gaze vector
    # color of the gaze vector based on the angle between
    # the gaze and the direction of the camera
    gaze = gaze / np.linalg.norm(gaze)
    gaze = torch.tensor(gaze)[None]
    target = torch.tensor([[0.0, 0.0, -1.0]])
    sim = F.cosine_similarity(gaze, target, dim=1, eps=1e-10)
    sim = F.hardtanh_(sim, min_val=-1.0, max_val=1.0)
    angle_gaze = torch.acos(sim)[0] * 180 / np.pi
    angle_gaze /= 180
    color = np.array(cmap(angle_gaze)[:3]) * 255
    image = draw_arrow2D(
        image=image,
        gaze=gaze[0],
        position=head_center,
        head_size=head_radius,
        d=0.05,
        color=color,
        thickness=thickness_gaze,
    )

    return image


# ========================= Main Class =========================== #


class Gaze3DDemo:
    """
    Gaze3DDemo class to predict gaze on videos or images.
    It first detects and tracks heads in the input video/image,
    then predicts gaze using the pre-trained model.
    Finally, it draws the predicted gaze on the input video/image and saves the output.

    It saves the detected/tracks heads and predicted gaze in the output dir as csv files.
    It also saves the output video/image with the predicted gaze drawn in the output dir.

    Args:
        input_filename (str): Name of the clip/image file to process (with extension).
        output_dir (str): Name of the folder where to save the output.
        ckpt_path (str): Path to the pre-trained model checkpoint.
        inference_modality (str): The modality inference to use could be image for video. (options: "image", "video")
        window_stride (int): only for video inference Stride used to create a clip. 1 seems to be provide more stable gaze.
        device (str): Device to use for inference. (options: "cpu", "cuda")
        batch_size (int): Batch size that can fit in RAM or GPU. (consider increasing for faster inference on video)
        num_workers (int): Number of worker for multiprocessing dataloader. (consider increasing for faster inference on video)

    Example:
        demo = Gaze3DDemo(
            input_filename="data/video.mp4",
            output_dir="output",
            ckpt_path="./checkpoints/gat_stwsge_gaze360_gf.ckpt",
            inference_modality="video",
            window_stride=1,
            device="cuda",
            batch_size=16,
            num_workers=4
        )
        demo.run() # detect, track, predict gaze and draw the output

        # optional
        demo.detect_and_track_heads() # detect and track heads
        demo.predict_gaze() # predict gaze
        demo.draw_prediction() # draw the predicted gaze on the input video/image

    """

    def __init__(
        self,
        input_filename: str,
        output_dir: str,
        ckpt_path: str,
        inference_modality: str = "image",
        window_stride: int = 1,
        device: str = "cpu",
        batch_size: int = 16,
        num_workers: int = 1,
    ):
        # Initialize arguments
        self.input_filename = input_filename
        self.output_dir = output_dir
        self.ckpt_path = ckpt_path
        self.modality = inference_modality
        self.window_stride = window_stride
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        # setup path
        self.setup()

        # Initialize models
        self.model = load_gaze_model(self.ckpt_path, self.device)
        self.head_detector = load_head_detection_model(self.device)
        self.tracker = load_tracker()

        # Initialize data structures
        self.detected_heads = []

    def setup(self):
        self.input_modality = identify_modality(self.input_filename)
        self.modality = "image" if self.input_modality == "image" else self.modality
        if self.modality == "video" and self.input_modality == "image":
            raise ValueError("Input modality is image but you ask for video model prediction.")

        self.input_name = os.path.basename(self.input_filename).split(".")[0]
        os.makedirs(self.output_dir, exist_ok=True)
        self.detected_head_file = os.path.join(
            self.output_dir, f"{self.input_name}_detected_head.csv"
        )
        self.predicted_gaze_file = os.path.join(
            self.output_dir, f"{self.input_name}_{self.modality}_predicted_gaze.csv"
        )
        self.output_file = (
            os.path.join(self.output_dir, f"{self.input_name}_{self.modality}_output.mp4")
            if self.input_modality == "video"
            else os.path.join(self.output_dir, f"{self.input_name}_{self.modality}_output.png")
        )

    def __save_detected_heads(
        self,
    ):
        df = pd.DataFrame(
            self.detected_heads, columns=["xmin", "ymin", "xmax", "ymax", "pid", "frame_id"]
        )
        df.to_csv(self.detected_head_file, index=False)

    def __detect_and_track_heads(self, image, frame_id):
        # 1. Convert image
        image_np = np.array(image)
        raw_detections = detect_heads(image_np, self.head_detector)
        detections = []
        for k, raw_detection in enumerate(raw_detections):
            bbox, conf = raw_detection[:4], raw_detection[4]
            if conf > DET_THR:
                cls_ = np.array([0.0])
                detection = np.concatenate([bbox, conf[None], cls_])
                detections.append(detection)
        detections = np.stack(detections)

        # 2. Detect & track head bboxes
        tracks = self.tracker.update(detections, image_np)
        if len(tracks) == 0:
            pass
        # pids = (tracks[:, 4] - 1).astype(int)
        # head_bboxes = torch.from_numpy(tracks[:, :4]).float()
        # add the frame_id to the tracks
        tracks = tracks[:, :5]
        tracks = np.hstack([tracks, np.ones((len(tracks), 1)) * frame_id])
        self.detected_heads.extend(tracks.tolist())

    def __process_video(
        self,
    ):
        # Read Video Clip
        cap = cv2.VideoCapture(self.self.input_filename)
        ret, frame = cap.read()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Iterate over frames and process
        frame_nb = 0
        with tqdm(total=frame_count, desc="Detect and track heads") as pbar:
            while ret:
                frame_nb += 1

                # =============== Predict =============== #
                frame_np = frame[..., ::-1]  # BGR >> RGB
                frame = Image.fromarray(frame_np)
                self.__detect_and_track_heads(frame, frame_nb)

                # =============== Read Next Frame =============== #
                ret, frame = cap.read()

                pbar.update(1)

        # Release Capture Device
        cap.release()

        # Save the detected heads
        self.__save_detected_heads()

    def __process_image(
        self,
    ):
        image = Image.open(self.input_filename)
        self.__detect_and_track_heads(image, 1)
        self.__save_detected_heads()

    def __draw_prediction_video(
        self,
    ):
        """
        load the video and the detected head bbox and draw the bbox on the video
        """

        # get the detected head bbox
        predicted_gaze = pd.read_csv(self.predicted_gaze_file)

        # Read Video Clip
        cap = cv2.VideoCapture(self.input_filename)
        ret, frame = cap.read()
        img_h, img_w, _ = frame.shape  # retrieve video height and width
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize ffmpeg writer
        command = f"ffmpeg -loglevel error -y -s {img_w}x{img_h} -pixel_format rgb24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -crf 24 {self.output_file}"
        command = shlex.split(command)
        process = sp.Popen(command, stdin=sp.PIPE)

        # Iterate over frames and process
        frame_nb = 0
        with tqdm(total=frame_count, desc="draw heads bbox") as pbar:
            while ret:
                frame_nb += 1

                # =============== Draw =============== #
                frame_np = frame[..., ::-1]  # BGR >> RGB
                frame = frame_np.copy().astype(np.uint8)
                for _, row in predicted_gaze[predicted_gaze["frame_id"] == frame_nb].iterrows():
                    head_box = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
                    head_pid = int(row["pid"])
                    gaze = np.array([row["gaze_x"], row["gaze_y"], row["gaze_z"]])
                    frame = draw_gaze(
                        frame,
                        head_box,
                        head_pid,
                        gaze,
                        CMAP,
                        COLORS,
                        thickness=10,
                        thickness_gaze=10,
                        fs=0.8,
                    )

                # ================= Write Frame ================= #
                process.stdin.write(frame.tobytes())

                # =============== Read Next Frame =============== #
                ret, frame = cap.read()

                pbar.update(1)

        # Release Capture Device
        cap.release()
        # Close and flush stdin
        process.stdin.close()
        # Wait for sub-process to finish
        process.wait()
        # Terminate the sub-process
        process.terminate()

    def __draw_prediction_image(self):
        """
        load the image and the detected head bbox and draw the bbox on the image
        """
        # get the detected head bbox
        predicted_gaze = pd.read_csv(self.predicted_gaze_file)
        image = cv2.imread(self.input_filename)

        # =============== Draw =============== #
        frame_np = image[..., ::-1]  # BGR >> RGB
        image = frame_np.copy().astype(np.uint8)

        for _, row in predicted_gaze.iterrows():
            head_box = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
            head_pid = int(row["pid"])
            gaze = np.array([row["gaze_x"], row["gaze_y"], row["gaze_z"]])
            image = draw_gaze(
                image,
                head_box,
                head_pid,
                gaze,
                CMAP,
                COLORS,
                thickness=10,
                thickness_gaze=10,
                fs=0.8,
            )

        # Save the image
        cv2.imwrite(self.output_file, image[..., ::-1])

    def detect_and_track_heads(
        self,
    ):
        self.detected_heads = []

        if self.input_modality == "video":
            self.__process_video()
        elif self.input_modality == "image":
            self.__process_image()

    def predict_gaze(
        self,
    ):
        if self.modality == "video":
            print("Predicting gaze using video prediction ...")
            dataset = DemoVideoData(
                detected_head_file=self.detected_head_file,
                input_file_path=self.input_filename,
                window_stride=self.window_stride,
            )
        elif self.modality == "image":
            print("Predicting gaze using image prediction ...")
            dataset = DemoImageData(
                detected_head_file=self.detected_head_file,
                input_file_path=self.input_filename,
            )
        else:
            raise ValueError("Unsupported input modality.")

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        gaze_stack = []
        # Iterate over the dataset
        for sample in tqdm(dataloader, desc="Predicting Gaze"):
            # Predict
            with torch.no_grad():
                pred = self.model(sample["images"].to(self.device))
                gaze = torch.nn.functional.normalize(pred["gaze"], p=2, dim=2, eps=1e-8)
                t = gaze.size(1)
                t = (
                    (t // 2) if t == 1 or t % 2 != 0 else (t // 2) - 1
                )  # defined middle frame in even case
                gaze_stack.append(gaze[:, t, :].cpu().numpy())
        # Save the gaze predictions
        gaze_stack = np.vstack(gaze_stack)
        detect_heads = pd.read_csv(self.detected_head_file)
        # Add the gaze predictions to the detected heads
        detect_heads["gaze_x"] = gaze_stack[:, 0]
        detect_heads["gaze_y"] = gaze_stack[:, 1]
        detect_heads["gaze_z"] = gaze_stack[:, 2]
        detect_heads.to_csv(self.predicted_gaze_file, index=False)

    def draw_prediction(self):
        if self.input_modality == "video":
            self.__draw_prediction_video()
        elif self.input_modality == "image":
            self.__draw_prediction_image()
        else:
            raise ValueError("Unsupported input modality.")

    def run(
        self,
    ):
        # Detect and track heads
        self.detect_and_track_heads()
        # Predict gaze
        self.predict_gaze()
        # Draw the detected heads
        self.draw_prediction()


if __name__ == "__main__":
    demo = Gaze3DDemo(
        input_filename=args.input_filename,
        output_dir=args.output_dir,
        ckpt_path=args.ckpt_path,
        inference_modality=args.modality,
        window_stride=args.window_stride,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    demo.run()
