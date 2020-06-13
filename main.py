# USAGE
# python main.py -i images/adrian.png

import argparse
import os
from pathlib import Path

import cv2

from src.detect_age import detect_age


if __name__ == "__main__":
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--image",
        required=True,
        help="Path to single image or directory of images",
    )
    ap.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.15,
        help="minimum probability to filter weak detections",
    )
    args = vars(ap.parse_args())

    # Set up models and networks as global variables

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = str(Path().cwd() / "models" / "face" / "deploy.prototxt")
    weightsPath = str(
        Path().cwd() / "models" / "face" / "res10_300x300_ssd_iter_140000.caffemodel"
    )
    faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

    print("[INFO] loading age detector model...")
    prototxtPath = str(Path().cwd() / "models" / "age" / "age_deploy.prototxt")
    weightsPath = str(Path().cwd() / "models" / "age" / "age_net.caffemodel")
    ageNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

    # Begin detection
    image = args["image"]
    if os.path.isdir(image):
        detect_age(image, args["confidence"], faceNet, ageNet)
    elif os.path.isfile(image):
        detect_age(
            image, min_confidence=args["confidence"], face_net=faceNet, age_net=ageNet
        )
    else:
        raise FileNotFoundError("Couldn't find directory or file.")
