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
        # Leave default here so we don't get some random
        # object as a face (i.e. sam jackson's eye at .15 min
        # confidence level).
        default=0.46,
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
        "--encoding-path",
        default="data/encoding.pickle",
        help="Output path for facial recognition encoding path.",
    )
    ap.add_argument(
        "--recognizer-path",
        default="data/recognizer.pickle",
        help="path to output model trained to recognize faces",
    )
    ap.add_argument(
        "--name-path",
        default="data/names.pickle",
        help="path to output label/name encoder",
    )
    return ap
