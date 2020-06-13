# USAGE
# python detect_age.py --image images/adrian.png

import argparse
import os
from pathlib import Path

import cv2

from src.detect_age import detect_age


faceNet = None
ageNet = None

# define the list of age buckets our age detector will predict
AGE_BUCKETS = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]

if __name__ == "__main__":
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--image",
        required=True,
        default="images/adrian.png",
        help="Path to single image or directory of images",
    )
    ap.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0,
        help="minimum probability to filter weak detections",
    )
    args = vars(ap.parse_args())

    # Set up models and networks as global variables

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = Path().cwd() / "models" / "face" / "deploy.prototxt"
    weightsPath = (
        Path().cwd() / "models" / "face" / "res10_300x300_ssd_iter_140000.caffemodel"
    )
    faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

    print("[INFO] loading age detector model...")
    prototxtPath = Path().cwd() / "models" / "age" / "age_deploy.prototxt"
    weightsPath = Path().cwd() / "models" / "age" / "age_net.caffemodel"
    ageNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

    if os.path.isdir(args["image"]):
        detect_age(args["image"], min_confidence=args["confidence"])
    elif os.path.isfile(args["image"]):
        detect_age(args["image"], min_confidence=args["confidence"])
    else:
        raise FileNotFoundError("Couldn't find directory or file.")
