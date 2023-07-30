# Faster-COCO-Eval

## Disclaimer

I often use this project, but I saw it abandoned and without a public repository on github.
Also, part of the project remained unfinished for a long time. I implemented some of the author's ideas and decided to make the results publicly available.

## Faster-COCO-Eval base

This package wraps a facebook C++ implementation of COCO-eval operations found in the
[pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) package.
This implementation greatly speeds up the evaluation time
for coco's AP metrics, especially when dealing with a high number of instances in an image.

## Comparison

For our use case with a test dataset of 5000 images from the coco val dataset.
Testing was carried out using the mmdetection framework and the eval_metric.py script. The indicators are presented below.

Visualization of testing **comparison.ipynb** available in directory [examples/comparison](./examples/comparison/comparison.ipynb)
Tested with yolo3 model (bbox eval) and yoloact model (segm eval)

| Type | COCOeval    | COCOeval_faster | Profit       |
| ---- | ----------- | --------------- | ------------ |
| bbox | 18.477 sec. | 7.345 sec.      | 2.5x faster  |
| segm | 29.819 sec. | 15.840 sec.     | 2x faster    |

## Usage

This package contains a faster implementation of the
 [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) `COCOEval` class.  
To import and use `COCOeval_faster` type:

````python  
from faster_coco_eval import COCO, COCOeval_faster
....
````

For usage, look at the original `COCOEval` [class documentation.](https://github.com/cocodataset/cocoapi)

## Usage plot curves

````python  
from faster_coco_eval import COCO
from faster_coco_eval.extra import Curves

cocoGt = COCO(....)
cocoDt = cocoGt.loadRes(....)

cur = Curves(cocoGt, cocoDt, iou_tresh=0.5, iouType='segm')
cur.plot_pre_rec(plotly_backend=False)
````

## Setup dependencies

- numpy
- plotly (optional if extra.Curve usage)  

## history

### v1.3.3

- [x] fix by ViTrox <https://github.com/vitrox-technologies/faster_coco_eval>
    - missing file issue
    - issue discovered by torchmetric
    - fstring for python3.7
    - Windows compilation

### v1.3.2

- [x] rework math_matches function. moved to faster_eval_api
- [x] Moved calculations from python to c++
- [x] Separated extra classes
- [x] Added new sample data
- [x] append mIoU based on TP pred.
- [x] append mAUC based on Coco pre/rec.

### v1.3.1

- [x] rework mask code
- [x] change np.float to float ([numpy deprecations](https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations))

### v1.3.0

- [x] remove pycocotools dependencies
- [x] clean c/c++ code

### v1.2.3

- [x] Implemented of mean IoU for TP
- [x] set FP-red FN-blue

### v1.2.2

- [x] Removed own implementation of pre-rec  
- [x] Switched to the implementation of pre-rec calculation from COCO eval  
- [x] Lost backward compatibility  
- [x] Implemented output fp/fn/tp + gt to pictures  

### v1.2.1

- [x] bug fix with pre-rec curve  
- [x] rework error calc (tp/fp/fn)  
- [x] change image plot to plotly
- [x] append docker auto builder  
- [x] append native iou calc (slow but accurate)  
- [x] rework auc calc with [link](https://towardsdatascience.com/how-to-efficiently-implement-area-under-precision-recall-curve-pr-auc-a85872fd7f14)  

### v1.1.3-v1.1.4

- [x] rebuild plotly backend
- [x] Segm bug-fix

### v1.1.2

- [x] Append fp fn error analysis via curves
- [x] Append confusion matrix
- [x] Append plotly backend support for ROC / AUC

### v1.1.1

- [x] Redesigned curves
- [x] Reworked data preload
- [x] Append csrc to setup
- [x] Build sdist Package

### v1.1.0

- [x] Wrap c++ code
- [x] Get it to compile
- [x] Add COCOEval class wraper
- [x] Remove detectron2 dependencies
- [x] Remove torch dependencies
- [x] Append unittest
- [x] Append ROC / AUC curves  
- [x] Check if it works on windows

### TODOs

- [X] Remove pycocotools dependencies
- [ ] Remove matplotlib dependencies

## License

The original module was licensed with apache 2, I will continue with the same license.
Distributed under the apache version 2.0 license, see [license](LICENSE) for more information.
