# Detecting Age with Static Images

Created 9 June 2020

Started AND Finished 13 June 2020

Published by Andrew Chen Wang

Table of Contents:
- Introduction
- How this works
- Technical details
- Conclusion + Benchmark
- What's left to be done
- References
- FAQ
- Contact Me
- License

---
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

Usage for multi-image dependant age estimation (you must have a dataset
in the data folder to use this with at least two people. Go to the
[Data README](https://github.com/Andrew-Chen-Wang/static-image-age-detection/blob/master/data/README.md)
for instructions on what you should have in the data directory):

```bash
python main.py -i data/input -d data/dataset
```

Usage for single image (i.e. not dependent on others, but you can specify a
directory, too): `python main.py --image data/single_images/adrian.png`

If you submit a directory of images, then we draw out the initial image
with bounding boxes. You can specify to show all images with bounding
boxes.

---
### Technical details

Besides the single image age detector, the multi-image age detector
utilizes several single image age estimations of a person and combines
them in an algorithm. You may notice there is a facial recognition
package; that is used for figuring out which person gets a certain age
bracket during the age detection calculation.

The facial recognition DOES NOT affect the results of the age estimation.
It is primarily designed to detect the person and log the correct age
bracket in case an input image has multiple people.

---
### Conclusion + Benchmark

Conclusion:

As of 13 June 2020 midnight, I finished it and tested it
on some female model that my friend sent me so that I can test this.
As noted the main [age-detection repo](https://github.com/Andrew-Chen-Wang/age-detection),
females are more difficult ot identify in age. The model turned out
to be in the 8-12 age bracket. I still need further testing though,
as I only did one.

In my opinion, it's just a matter of the age detector model used.
I cannot be conclusive about if multiple images help or not,
so I'm going to try and find a better age detector model.

Benchmark:

The reason I wanted to do this was to improve the accuracy of age detection
using videos. To put it to the test, you can run `benchmark.py` which
will compare the accuracy of just doing a single image test (for an
entire directory if specified) vs. processing an entire directory.
The single image test takes the mean of the accuracies to show
the stability of using multiple images for processing instead of one.

Have I looked through all of OpenCV or PyImageSearch? Nope, only one
article. But hey, beginner's luck I suppose.

---
### What's left to be done

Again, my goal was to test this aggregate data for video streaming
purposes. Although my purpose of video streaming is to tag people
and not to train on known people, the concept remains the same
as the training doesn't affect the age detection (we know the age
detection must work if the face detector detects the target person).

So really, for this repo, it's making the aggregate data function
(ambiguously) better. For me, it's to learn how to train to detect
an unknown person and tag him/her with an ID.

The last thing that needs work, for me, is when a new picture comes
in. Perhaps at an 87% confidence threshold can you retrain a embedding
using this new dataset (that just includes that one new photo). It's
similar to how social media companies do it, but it'd be neat to learn!

---
### FAQ

**Why am I getting** `ValueError: The number of classes has to be greater than one; got 1 class`?

You should have at least two different, known people in your dataset directory.

---
### References

The code and models largely come Adrian at PyImageSearch.
This is my first time learning machine learning and deep
learning (not ready for AI yet).

Luckily, in this tutorial, Adrian used DDN and I tweaked a
bit of code to just be more efficient. Of course, this project
is not all about one image but multiple to increase accuracy,
which is my code.

I'm grateful for finding this because I couldn't find the models
myself lol. So thanks for the tutorials, and I hope this
module actually DOES increase accuracy while maintaining
a good CPU usage level (slight jab at Dlib's HoG).

Additionally, I used a lot, like a lot, of code from
Adrian's tutorials, and only tweaked them to be more
modern and lean (e.g. pathlib, unnecessary args, personal
modifications to suit the project), so a big shout out
to him for his wonderful work!

---
### Contact Me

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
