import cv2
import numpy as np


def detect_age(image, min_confidence: float, show_image: bool = False):
    global faceNet
    global ageNet
    # load the input image and construct an input blob for the image
    image = cv2.imread(image)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    faceNet.setInput(blob)
    detections = faceNet.forward()

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
                face,
                1.0,
                (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False,
            )

            # make predictions on the age and find the age bucket with
            # the largest corresponding probability
            ageNet.setInput(faceBlob)
            preds = ageNet.forward()
            i = preds[0].argmax()
            age = AGE_BUCKETS[i]
            ageConfidence = preds[0][i]

            # display the predicted age to our terminal
            text = "{}: {:.2f}%".format(age, ageConfidence * 100)
            print(f"[INFO] {text}")

            if show_image:
                # draw the bounding box of the face along with the associated
                # predicted age
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(
                    image,
                    text,
                    (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    2,
                )

    if show_image:
        # display the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
