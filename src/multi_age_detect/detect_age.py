"""
This age detector is different from the single image
age detector in that we're, in case there are multiple
people in the image, we can figure out which person
belongs to which outputted age bracket + confidence.
"""

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
