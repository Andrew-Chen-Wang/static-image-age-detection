# Detecting Age with Static Images

Created 9 June 2020

### Introduction

The purpose of this package is to identify age through
static images. Following some four repositories later
and not knowing much machine learning techniques (only
concepts), I decided to use OpenCV's DNN to better
accommodate for future plans of live calls in addition
to some other models in order to deal with dropped FPS.

For now, this is all about static images, finding out
how many images are needed to get an average accurate
prediction rate of 95% and finding ways to improve
the rate while decreasing the number of images needed
for proper identification.

Note: when speaking of utilizing multiple images, it's
to increase accuracy by getting more features at different
angles and curves in case that's the reason.

---
### How this works

This module uses the tutorial by PyImageSearch seen 
[here](https://www.pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/).
I've reworked this code to utilize a single image or a directory
of images so that I can increase accuracy.

Really, the increase of accuracy comes from the mean age combined
with the intertwining of the confidence of each age bracket. The
directory usage has an initial image (which you can specify in
an argument if it's not the first in alphanumeric order) which
we use to define the initial objects. There must be at least one
object that appears in all submitted pictures.

I've also integrated object tagging so that I know that the objects
are the same; if not, then we raise an error saying they are different.
If there are multiple objects in the initial image, we only raise an
error if none of the initial faces appear in the initial picture.

Usage: `python detect_age.py --image images/adrian.png`

If you submit a directory of images, then we draw out the initial image
with bounding boxes. You can specify to show all images with bounding
boxes.

---
### References

The code and models largely come Adrian at PyImageSearch.
This is my first time learning machine learning and deep
learning (not ready for AI yet).

Luckily, in this tutorial, Adrian used DDN and I tweaked a
bit of code to just be more efficient. Of course, this project
is not all about one image but multiple to increase accuracy,
which is my code.

I'm grateful for finding this because I couldn't find them
myself lol. So thanks for the tutorials, and I hope this
module actually DOES increase accuracy while maintaining
a good CPU usage level (slight jab at Dlib's HoG).

---
### Contact

If I'm doing something wrong (again, I've never learned machine
learning before. I only know some basic concepts and ideas),
please open a GitHub issue or contact me here:

[Andrew Wang at acwangpython@gmail.com](mailto:acwangpython@gmail.com?subject=[GH%20DL%20Static%20Image])

(If you decide to email me, please leave the prefix subject line. Thank you).

---
### License
```
   Copyright 2020 Andrew Chen Wang

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
