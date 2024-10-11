# history

## v1.6.1 - v1.6.3

- [x] fix windows & macos build for [torchmetrics](https://github.com/Lightning-AI/torchmetrics/pull/2750)

## v1.6.0

- [x] Rework [mask_api](csrc/mask_api/mask_api.cpp) with pybind11 C++ .
- [x] Rework [RLE support](csrc/mask_api/src/rle.cpp).
- [x] Create test files for all components.
- [x] The math_matches function has been reworked, with an emphasis on using C++ code.
- [x] Added more documentation of functions. Optimized existing ones.
- [x] Added rleToBoundary func with 2 backend \["mask_api", "opencv"\]
- [x] IoU type [boundary](https://github.com/bowenc0221/boundary-iou-api/tree/master) support (further testing is needed)
- [x] Create async rle and boundary comput [discussion](https://github.com/MiXaiLL76/faster_coco_eval/pull/31#issuecomment-2308369319)

## v1.5.7

- [x] Compare COCOEval bug fix.

## v1.5.6

- [x] Replace CED MSE curve with MAE (px) curve
- [x] Add CED examples
- [x] Display IoU and MAE for keypoints
- [x] Reworked eval.\_prepare to clear up the return flow
- [x] Reworked the C++ part of **COCOevalEvaluateImages** and **COCOevalAccumulate**
  - [x] Add new **COCOevalEvaluateAccumulate** to combine these two calls. You can use old style **separate_eval==True** (default=False)
  - [x] **COCOevalAccumulate** & **COCOevalEvaluateAccumulate** -> *COCOeval_faster.eval* is now correctly created as numpy arrays.
- [x] Append LVIS dataset support **lvis_style=True** in COCOeval_faster

```py
cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType, lvis_style=True, print_function=print)
cocoEval.params.maxDets = [300]
```

## v1.5.5

- [x] Add CED MSE curve
- [x] Review tests
- [x] Review **COCOeval_faster.math_matches** function and **COCOeval_faster.compute_mIoU** function

## v1.5.3 - v1.5.4

- [x] Worked out the ability to work with skeletons and various key points
- [x] `eval.state_as_dict` Now works for key points

## v1.5.2

- [x] Change comparison to colab_example
- [x] append utils with opencv **conver_mask_to_poly** (extra)
- [x] append **drop_cocodt_by_score** for extra eval

## v1.5.1

- [x] **breaking change** | new static function COCO.load_json
- [x] new curve f1_confidence with `cur.plot_f1_confidence()`
- [x] **breaking change** | replace display_matrix arg `in_percent` to `normalize`
- [x] **breaking change** | rework draw functions

## v1.4.2

- [x] append Auto-formatters
- [x] append py36 support
- [x] append pandas to requirements for plotly\[express\]
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
