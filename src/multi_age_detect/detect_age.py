"""
This age detector is different from the single image
age detector in that we're, in case there are multiple
people in the image, we can figure out which person
belongs to which outputted age bracket + confidence.
"""
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .face_recognition import face_recognizer


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


def detect_age(
    image,
    min_confidence: float,
    face_net,
    age_net,
    embedder,
    recognizer_path,
    name_path,
    face_prob: float,
    show_image: bool = False,
) -> Dict[str : List[Tuple[float, str, float]]]:
    # load the input image and construct an input blob for the image
    image = cv2.imread(image)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image=image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0)
    )

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    face_net.setInput(blob)
    detections = face_net.forward()

    # Stores the age bracket and confidence for each detection.
    brackets = {}

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > min_confidence:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the ROI of the face and then construct a blob from
            # *only* the face ROI
            face = image[startY:endY, startX:endX]
            faceBlob = cv2.dnn.blobFromImage(
                image=face,
                scalefactor=1.0,
                size=(227, 227),
                mean=(78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False,
            )

            # make predictions on the age and find the age bucket with
            # the largest corresponding probability
            age_net.setInput(faceBlob)
            preds = age_net.forward()
            i = preds[0].argmax()
            age = AGE_BUCKETS[i]
            ageConfidence = preds[0][i]

            # display the predicted age to our terminal
            text = "{}: {:.2f}%".format(age, ageConfidence * 100)
            print(f"[INFO] {text}")

            # try to recognize the face for the aggregation formula
            name, prob = face_recognizer(
                embedder, recognizer_path, name_path, face, face_prob
            )
            brackets[name].append((prob, age, ageConfidence * 100))

            if show_image:
                # draw the bounding box of the face along with the associated
                # predicted age
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(
                    img=image,
                    pt1=(startX, startY),
                    pt2=(endX, endY),
                    color=(0, 0, 255),
                    thickness=2,
                )
                cv2.putText(
                    img=image,
                    text=text,
                    org=(startX, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.45,
                    color=(0, 0, 255),
                    thickness=2,
                )

    if show_image:
        # display the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    return brackets
