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

Visualization of testing **comparison.ipynb** available in directory [examples/comparison](https://github.com/MiXaiLL76/faster_coco_eval/blob/main/examples/comparison/comparison.ipynb)
Tested with yolo3 model (bbox eval) and yoloact model (segm eval)

| Type | COCOeval    | COCOeval_faster | Profit      |
| ---- | ----------- | --------------- | ----------- |
| bbox | 18.477 sec. | 7.345 sec.      | 2.5x faster |
| segm | 29.819 sec. | 15.840 sec.     | 2x faster   |

## Feautures

This library provides not only validation functions, but also error visualization functions. Including visualization of errors in the image.  
You can study in more detail in the [examples](https://github.com/MiXaiLL76/faster_coco_eval/blob/main/examples) and [Wiki](https://github.com/MiXaiLL76/faster_coco_eval/wiki).

## Usage

Code examples for using the library are available on the [Wiki](https://github.com/MiXaiLL76/faster_coco_eval/wiki)

## Update history

Available via link [history.md](https://github.com/MiXaiLL76/faster_coco_eval/blob/main/history.md)

## License

The original module was licensed with apache 2, I will continue with the same license.
Distributed under the apache version 2.0 license, see [license](LICENSE) for more information.
