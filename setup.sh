#!/bin/bash

# Create weights directory if it doesn't exist
mkdir -p weights

# Download the head detector weights
# https://github.com/MahenderAutonomo/yolov5-crowdhuman/blob/master/weights/download_weights.sh
gdown 'https://drive.google.com/uc?id=1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb' -O ./weights/crowdhuman_yolov5m.pt

echo "Download complete: weights/crowdhuman_yolov5m.pt"