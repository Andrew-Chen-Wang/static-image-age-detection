"""
Single Image Usage
python main.py -i data/single_images/adrian.png

Multiple-Aggregate Image Usage
python main.py -i data/multi_images/barack -d=data/dataset --encoding-path=data/encoding.pickle
"""

import os
from pathlib import Path

import cv2

from src.arg_builder import build_args
from src.multi_age_detect.prepare.encode_faces import encode_faces
from src.single_image.detect_age import detect_age as single_image_age_detect


if __name__ == "__main__":
    args = vars(build_args().parse_args())

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
        if args["without_aggregate"]:
            # This simply gets the age of each person independent of other images
            for filename in os.listdir(image):
                single_image_age_detect(
                    filename, args["confidence"], faceNet, ageNet, args["show_first"]
                )
        else:
            encode_faces(
                dataset_path=args["dataset"], encoding_path=args["encoding_path"]
            )

    elif os.path.isfile(image):
        single_image_age_detect(
            image,
            min_confidence=args["confidence"],
            face_net=faceNet,
            age_net=ageNet,
            show_image=args["show_first"],
        )
    else:
        raise FileNotFoundError("Couldn't find directory or file.")
