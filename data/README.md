# Data Directory

This is where the data directory lives to
include paths for images that could be used
for training the facial recognition,
storing input photos, etc.

There is a `.gitignore` file so that encodings,
your dataset, and some other large data
related files aren't being added to git.

Folders:
- `dataset` is for the images that will be
  used for the encoding data (which is used for
  facial recognition in further usage of the age
  estimator for unprocessed images).
  - You MUST have two people in the dataset for this 
    to work.
  - The format of the `dataset` folder is that there
    is a subdirectory with the person's name. 
    In those directories, add the images of that 
    person. PLEASE make sure that there is only one 
    person in each photo, and increase the confidence 
    threshold if you must.
  - These images are NOT used for the age estimation.
    They are only used for facial recognition so that
    the multi-image age detector can figure out which
    person gets which age bracket.
  - Note: This is NOT added to git as it's too large
    of a dir. I've specified this dir in the gitignore.
- `single_images` is for simple single-image
  age estimation, independent of any other image
  processed before.

User Folders:
- `input` is for the images that you'd like to
  put into the age estimator. This is mostly used
  for the multi-image age estimator which means
  the images are dependent on each other for
  estimating the age.
  - Note: The input images for estimating age are
    NOT dependent on the encoding images. Those encoding
    images are used for facial recognition.
