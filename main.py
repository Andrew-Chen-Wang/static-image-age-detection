"""
Single Image Usage
python main.py -i data/single_images/adrian.png

Multiple-Aggregate Image Usage
python main.py -i data/multi_images/barack -d=data/dataset --encoding-path=data/encoding.pickle
"""

import os
from pathlib import Path

import cv2
from click import confirm

from src.arg_builder import build_args
from src.multi_age_detect.prepare.encode_faces import encode_faces
from src.multi_age_detect.prepare.train import create_recognizer_with_label
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

    print("[INFO] loading face recognizer model...")
    embedderModel = str(
        Path().cwd() / "models" / "recognizer" / "openface_nn4.small2.v1.t7"
    )
    # Used for the recongition part, not for the training.
    embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

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
            # Preparing the data and models
            assert None not in (
                args["dataset"],
                args["encoding_path"],
                args["recognizer_path"],
                args["name_path"],
            ), "You must specify the path to the dataset directory and newly created encoding path"

            if os.path.isfile(args["encoding_path"]):
                choice = confirm(
                    "Encoding file already exists. Overwrite?", default=False
                )
                if choice:
                    encode_faces(args["dataset"], args["encoding_path"])
            else:
                encode_faces(
                    dataset_path=args["dataset"], encoding_path=args["encoding_path"]
                )

            # Create the recognizer and name pickles.
            if os.path.isfile(args["recognizer_path"]) or os.path.isfile(
                args["name_path"]
            ):
                choice = confirm(
                    "Recognizer/name encoder already exists. Overwrite?", default=False
                )
                if choice:
                    create_recognizer_with_label(
                        args["encoding_path"],
                        args["recognizer_path"],
                        args["name_path"],
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
