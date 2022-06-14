# Disclaimer
I often use this project, but I saw it abandoned and without a public repository on github.
Also, part of the project remained unfinished for a long time. I implemented some of the author's ideas and decided to make the results publicly available.

# Faster-COCO-Eval 
This package wraps a facebook C++ implementation of COCO-eval operations found in the 
[pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) package.
This implementation greatly speeds up the evaluation time
for coco's AP metrics, especially when dealing with a high number of instances in an image.

### Comparison

For our use case with a test dataset of 1500 images that contains up to 2000 instances per image we saw up to a 100x faster 
evaluation using faster-coco-eval (FCE) compared to the original pycocotools code.
````
Seg eval pycocotools 4 hours 
Seg eval FCE: 2.5 min

BBox eval pycocotools: 4 hours 
BBox eval FCE: 2 min
````

# Getting started

### Install
Available packages for download will be generated later

## Usage

This package contains a faster implementation of the 
[pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) `COCOEval` class.  
To import and use `COCOeval_faster` type:

````python
from faster_coco_eval import COCOeval_faster
````

For usage, look at the original `COCOEval` [class documentation.](https://github.com/cocodataset/cocoapi)

### Dependencies
- pycocotools
- pybind11
- numpy

# TODOs
- [x] Wrap c++ code
- [x] Get it to compile
- [x] Add COCOEval class wraper
- [x] Remove detectron2 dependencies
- [x] Remove torch dependencies
- [ ] Check if it works on windows

# License
The original module was licensed with apache 2, I will continue with the same license.
Distributed under the apache version 2.0 license, see [license](LICENSE) for more information.