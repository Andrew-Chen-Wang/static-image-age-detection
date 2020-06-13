"""
The purpose of face recognition between these
photos is for knowing which person we're using
from each image so that the aggregation of data
is properly processed.

This does NOT affect the age estimation of input
images by the user.
"""
import pickle
from typing import Tuple

import numpy as np

# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
# image = cv2.imread(args["image"])
# image = imutils.resize(image, width=600)
# (h, w) = image.shape[:2]


def face_recognizer(
    embedder, recognizer_path, names_path, face_blob
) -> Tuple[str, float]:
    """
    Assuming a face has been detected, try
    to recognize who the person is.
    """
    # construct a blob from the image
    # imageBlob = cv2.dnn.blobFromImage(
    #     cv2.resize(image, (300, 300)), 1.0, (300, 300),
    #     (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # extract the face ROI
    # face = image[startY:endY, startX:endX]
    # (fH, fW) = face.shape[:2]

    # ensure the face width and height are sufficiently large
    # if fW < 20 or fH < 20:
    #     continue
    # construct a blob for the face ROI, then pass the blob
    # through our face embedding model to obtain the 128-d
    # quantification of the face
    # faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
    #                                  (0, 0, 0), swapRB=True, crop=False)

    embedder.setInput(face_blob)
    vec = embedder.forward()

    # Load the pickles
    recognizer = pickle.loads(open(recognizer_path, "rb").read())
    le = pickle.loads(open(names_path, "rb").read())

    # perform classification to recognize the face
    predictions = recognizer.predict_proba(vec)[0]
    j = np.argmax(predictions)
    probability = predictions[j]
    name = le.classes_[j]
    return name, probability * 100
