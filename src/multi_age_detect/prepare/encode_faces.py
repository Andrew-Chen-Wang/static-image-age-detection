"""
Purpose of this is to create an encoding file so that
the facial recognition during the age detection can
know for sure which estimation goes with who in an
image. It all culminates into the final age estimation
algorithm which requires that we know which data point
goes with which face.
"""
import os
import pickle

import cv2
import face_recognition

from ...utils.list_images import list_images


def encode_faces(dataset_path: str, encoding_path: str) -> None:
    """
    Creates an encoding file for facial recognition.

    This is used for training our face and name/label model.
    The given model in models/recognizer/openface_nn4.small2.v1.t7
    is used for the face recognizer function itself.

    :param dataset_path: The path to the input dataset
    used for creating the encoding. Dataset should include
    faces and images of one person.
    :param encoding_path: The path where the encoding
    will be outputted in a pickle file.
    :return: None
    """
    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(list_images(dataset_path))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # loop over the image paths
    for i, imagePath in enumerate(imagePaths):
        # extract the person name from the image path
        print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model="cnn")
        # We use model cnn, not HoG, for accuracy

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}

    if not os.path.isabs(encoding_path):
        from pathlib import Path

        # Root Dir is NOT src but where main.py is.
        encoding_path = str(
            Path(__file__).resolve(strict=True).parent.parent.parent.parent
            / encoding_path
        )
    f = open(encoding_path, "wb")
    f.write(pickle.dumps(data))
    f.close()
