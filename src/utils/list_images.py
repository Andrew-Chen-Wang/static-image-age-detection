import os
from typing import List


def list_images(base_path: str) -> List[str]:
    """
    Lists all image paths in specified directory.
    Taken mostly from imutils with some adds/subtracts:
    https://github.com/jrosebr1/imutils/blob/master/imutils/paths.py
    """
    image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    for (rootDir, dirNames, filenames) in os.walk(base_path):
        # loop over the filenames in the current directory
        for filename in filenames:
            # check to see if the file is an image and should be processed
            if os.path.splitext(filename)[1].lower() in image_types:
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath
