
# history

## v1.4.3
- [x] fix [issue](https://github.com/MiXaiLL76/faster_coco_eval/issues/19)

## v1.4.2
- [x] append Auto-formatters 
- [x] append py36 support
- [x] append pandas to requirements for plotly[express]
- [x] update mask api with pycootools

## v1.4.1

- [x] append Plotly fig return 
- [x] append preview GT only func. Without eval.

```py
cocoGt = COCO(...)
preview = PreviewResults(cocoGt, iouType='segm')
preview.display_tp_fp_fn(data_folder=..., image_ids=..., display_gt=True)
```

## v1.4.0

- [x] fix issue <https://github.com/MiXaiLL76/faster_coco_eval/issues/12>
- [x] Updated pre-rec calculation method
- [x] Updated required libraries
- [x] Moved all matplotlib dependencies to plotly
- [x] Append new examples & mmeval test file

## v1.3.3

- [x] fix by ViTrox <https://github.com/vitrox-technologies/faster_coco_eval>
    - missing file issue
    - issue discovered by torchmetric
    - fstring for python3.7
    - Windows compilation

## v1.3.2

- [x] rework math_matches function. moved to faster_eval_api
- [x] Moved calculations from python to c++
- [x] Separated extra classes
- [x] Added new sample data
- [x] append mIoU based on TP pred.
- [x] append mAUC based on Coco pre/rec.

## v1.3.1

- [x] rework mask code
- [x] change np.float to float ([numpy deprecations](https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations))

## v1.3.0

- [x] remove pycocotools dependencies
- [x] clean c/c++ code

## v1.2.3

- [x] Implemented of mean IoU for TP
- [x] set FP-red FN-blue

## v1.2.2

- [x] Removed own implementation of pre-rec  
- [x] Switched to the implementation of pre-rec calculation from COCO eval  
- [x] Lost backward compatibility  
- [x] Implemented output fp/fn/tp + gt to pictures  

## v1.2.1

- [x] bug fix with pre-rec curve  
- [x] rework error calc (tp/fp/fn)  
- [x] change image plot to plotly
- [x] append docker auto builder  
- [x] append native iou calc (slow but accurate)  
- [x] rework auc calc with [link](https://towardsdatascience.com/how-to-efficiently-implement-area-under-precision-recall-curve-pr-auc-a85872fd7f14)  

## v1.1.3-v1.1.4

- [x] rebuild plotly backend
- [x] Segm bug-fix

## v1.1.2

- [x] Append fp fn error analysis via curves
- [x] Append confusion matrix
- [x] Append plotly backend support for ROC / AUC

## v1.1.1

- [x] Redesigned curves
- [x] Reworked data preload
- [x] Append csrc to setup
- [x] Build sdist Package

## v1.1.0

- [x] Wrap c++ code
- [x] Get it to compile
- [x] Add COCOEval class wraper
- [x] Remove detectron2 dependencies
- [x] Remove torch dependencies
- [x] Append unittest
- [x] Append ROC / AUC curves  
- [x] Check if it works on windows
