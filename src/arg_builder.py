"""
This is where all the arguments for this module lives.
"""
from argparse import ArgumentParser


def build_args() -> ArgumentParser:
    # Construct the argument parse and parse the arguments
    ap = ArgumentParser()
    ap.add_argument(
        "-i",
        "--image",
        required=True,
        help="Path to single image or directory of images.",
    )
    ap.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.15,
        help="Minimum probability to filter weak detections.",
    )

    # Multi-image age estimation args
    ap.add_argument(
        "-sf",
        "--show-first",
        action="store_true",
        help="Shows the first image after processing.",
    )
    ap.add_argument(
        "-wa",
        "--without-aggregate",
        action="store_true",
        help="If a directory is specified, then process without aggregation of data.",
    )
    ap.add_argument(
        "-d",
        "--dataset",
        help="Specifies the dataset to use for training the facial recognition. "
        "Used for multi-image age estimation so that the correct person is used "
        "during calculations.",
    )
    ap.add_argument(
        "--encoding-path", help="Output path for facial recognition encoding path.",
    )

    return ap
