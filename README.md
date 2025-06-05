# [CVPR 2025] Enhancing 3D Gaze Estimation in the Wild using Weak Supervision with Gaze Following Labels
| ![Demo 1](./assets/video_output_1.gif) | ![Demo 2](./assets/video_output_2.gif) |
|--------------------------------------|--------------------------------------|

**Note:** The color of the gaze vector follows a gradient from blue (frontal gaze toward the camera) to green (gaze directed away from the camera). Red indicates a gaze perpendicular to the camera, appearing in the middle of the gradient.

## Overview 

**Authors:** Pierre Vuillecard, Jean-marc Odobez 

[`Paper`](https://publications.idiap.ch/publications/show/5585) | [`BibTeX`](#citation)

This repository contains the code and checkpoints for our CVPR 2025 paper "Enhancing 3D Gaze Estimation in the Wild using Weak Supervision with Gaze Following Labels". 

## Setup & Installation

First, we need to clone the repository 
```shell
git clone 
cd <name_of_the_repo>
```

Next, create the conda environment and activate it after installing the necessary packages:
```shell
conda env create -f environment.yaml
conda activate gazeCVPR
```

Download model for head detection:
```shell
bash setup.sh
```

## Run the demos

The demo code is located in the demo.py file. Our Gaze Transformer model can perform inference on both images and videos. The demo first detects all the heads in the input using a head detector. Then, for each detected head, it predicts the gaze direction using our Gaze Transformer model. The output is drawn on the image or video and saved to the specified output directory.

To run the demo on images, only image inference is availabale: 
```python
python demo.py --input-filename data/pexels-jopwell-2422290.jpg --output-dir output/ --modality image
```

To run the demo on videos, you can either run the model with image inference where each frame is processed independently: 
```python
python demo.py --input-filename data/7149282-hd_1280_720_25fps.mp4 --output-dir output/ --modality image
```

Or you can run the model with video inference where the model uses temporal information to process the video. 
```python
python demo.py --input-filename data/7149282-hd_1280_720_25fps.mp4 --output-dir output/ --modality video
```

Additional parameters can be set such as the model checkpoint
```python
--checkpoint ./checkpoints/gat_stwsge_gaze360_gf.ckpt # best SOTA model trained on Gaze360 and Gazefollow
--checkpoint ./checkpoints/gat_gaze360.ckpt # best model trained on Gaze360 only
```

You can also run the demo on GPU by adding the `--device cuda`. To reduce the inference time on video a batch processing is performed. Therefore, the `--batch-size` and `--num-workers` parameters can be set to speed up inference.

## Training

The dataset preprocessing for training will be released in the future.


## License & Third-party resources

### Dataset used for training:
 - [Gaze360](https://github.com/erkil1452/gaze360/blob/master/LICENSE.md) License: Research use only
 - [Gazefollow](http://gazefollow.csail.mit.edu/) License: NonCommercial 

### Code 
 - Our model is buit upon Swin3D from omnivore and used as pretrained model. [code](https://github.com/facebookresearch/omnivore) License: Attribution-NonCommercial 4.0 International

### Demo: 
 - Tracking of face bounding box Boxmot [code](https://github.com/mikel-brostrom/boxmot) License: AGPL-3.0
 - Head detector from [code](https://github.com/MahenderAutonomo/yolov5-crowdhuman) Licence: GPL-3.0 using yolov5 (Licence : AGPL-3.0) [code](https://github.com/ultralytics/yolov5/tree/master?tab=AGPL-3.0-1-ov-file) trained on crowdhuman dataset [data](https://www.crowdhuman.org/) (Licence: NonCommercial)

## Citation

If you use this work, please cite the following paper:

```
@INPROCEEDINGS{vuillecard3DGAZEWILD2025,
         author = {Vuillecard, Pierre and Odobez, Jean-Marc},
       projects = {Idiap},
          month = jun,
          title = {Enhancing 3D Gaze Estimation in the Wild using Weak Supervision with Gaze Following Labels},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
           year = {2025},
            pdf = {https://publications.idiap.ch/attachments/papers/2025/Pierre_3DGAZEESTIMATIONINTHEWILD_2025.pdf}
}
```