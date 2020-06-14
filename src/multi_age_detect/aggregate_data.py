"""
Takes the confidence and age bracket of each image
and performs calculations to determine final age
bracket with an aggregate confidence.
"""
import numpy as np


# FIXME So this definitely needs some improving on
#  The algorithm can try to use more parameters.
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


def calculate_age(age_brackets: list) -> str:
    """
    Calculates each person's age bracket based on
    the person's given parameters.

    The parameters are in an array of tuples with
    array of tuples:
        index 0: probability of this person being... uh a known person
        index 1: age bracket as a string
        index 2: confidence in age detection as a density curve as floats

    This algorithm tries to use the confidence in the
    OpenCV person detection and

    :param age_brackets: the list of parameter tuples for this one person.
    :return: age bracket of person
    """
    confidence_intervals = np.sum(
        [age_brackets[x][2] for x in range(len(age_brackets))]
    )
    return AGE_BUCKETS[confidence_intervals.argmax()]
