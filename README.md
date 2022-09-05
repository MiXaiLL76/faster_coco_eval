# Disclaimer
I often use this project, but I saw it abandoned and without a public repository on github.
Also, part of the project remained unfinished for a long time. I implemented some of the author's ideas and decided to make the results publicly available.

# Faster-COCO-Eval 
This package wraps a facebook C++ implementation of COCO-eval operations found in the 
[pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) package.
This implementation greatly speeds up the evaluation time
for coco's AP metrics, especially when dealing with a high number of instances in an image.

### Comparison

For our use case with a test dataset of 5000 images from the coco val dataset.
Testing was carried out using the mmdetection framework and the eval_metric.py script. The indicators are presented below.

Visualization of testing **comparison.ipynb** available in directory [examples/comparison](./examples/comparison/comparison.ipynb)
Tested with yolo3 model (bbox eval) and yoloact model (segm eval)

Type | COCOeval | COCOeval_faster | Profit
-----|----------|---------------- | ------
bbox | 22.854 sec. | 8.714 sec.   | more than 2x 
segm | 35.356 sec. | 18.403 sec.  | 2x


# Getting started

## Local build
Build wheel
```bash
python3 setup.py bdist_wheel
```

Build wheel from docker
```bash
export PYTHON3_VERSION=3.9.10
docker build -f ./Dockerfile --build-arg PYTHON3_VERSION=${PYTHON3_VERSION} --tag faster_coco_eval:${PYTHON3_VERSION} .
docker run -v $(pwd):/app/src faster_coco_eval:${PYTHON3_VERSION}
```

## Install
Install form source  
```bash  
pip3 install git+https://github.com/MiXaiLL76/faster_coco_eval  
```  

## Usage

This package contains a faster implementation of the 
[pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) `COCOEval` class.  
To import and use `COCOeval_faster` type:

````python  
from faster_coco_eval import COCO, COCOeval_faster
....
````

For usage, look at the original `COCOEval` [class documentation.](https://github.com/cocodataset/cocoapi)

### Dependencies
- pycocotools
- pybind11
- numpy

# v1.1.0
- [x] Wrap c++ code
- [x] Get it to compile
- [x] Add COCOEval class wraper
- [x] Remove detectron2 dependencies
- [x] Remove torch dependencies
- [x] Append unittest
- [x] Append ROC / AUC curves  
- [x] Check if it works on windows

# v1.1.1
- [x] Redesigned curves
- [x] Reworked data preload
- [x] Append csrc to setup
- [x] Build sdist Package

# v1.1.2
- [x] Append fp fn error analysis
- [x] Append confusion matrix
- [x] Append plotly backend support for ROC / AUC

# v1.1.3
- Segm bug-fix

# v1.1.4
- rebuild plotly backend

# v1.1.5
- bug fix

# v1.2.1
- bug fix with pre-rec curve  
- rework error calc (tp/fp/fn)  
- change image plot to plotly   
- append docker auto builder  
- append native iou calc (slow but accurate)  
- rework auc calc with [link](https://towardsdatascience.com/how-to-efficiently-implement-area-under-precision-recall-curve-pr-auc-a85872fd7f14)  

# TODOs
- [ ] Remove pycocotools dependencies

# License
The original module was licensed with apache 2, I will continue with the same license.
Distributed under the apache version 2.0 license, see [license](LICENSE) for more information.
