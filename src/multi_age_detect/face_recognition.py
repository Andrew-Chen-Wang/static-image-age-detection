"""
The purpose of face recognition between these
photos is for knowing which person we're using
from each image so that the aggregation of data
is properly processed.

This does NOT affect the age estimation of input
images by the user.
"""
import pickle
from uuid import uuid4
from typing import Tuple

import cv2
import numpy as np


def face_recognizer(
    embedder, recognizer_path, names_path, face, min_prob: float = 0.4
) -> Tuple[str, float]:
    """
    Assuming a face has been detected, try
    to recognize who the person is.
    """
    (fH, fW) = face.shape[:2]

    assert fW > 20 or fH > 20, "Ensure the face width and height are sufficiently large"
    # construct a new blob for the face ROI, then pass the blob
    # through our face embedding model to obtain the 128-d
    # quantification of the face
    face_blob = cv2.dnn.blobFromImage(
        face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
    )

    embedder.setInput(face_blob)
    vec = embedder.forward()

    # Load the pickles
    recognizer = pickle.loads(open(recognizer_path, "rb").read())
    le = pickle.loads(open(names_path, "rb").read())

    # perform classification to recognize the face
    predictions = recognizer.predict_proba(vec)[0]
    j = np.argmax(predictions)
    probability = predictions[j]
    if probability < min_prob:
        return str(uuid4()), probability * 100
    else:
        return le.classes_[j], probability * 100
