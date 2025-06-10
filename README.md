# Faster-COCO-Eval

[![PyPI](https://img.shields.io/pypi/v/faster-coco-eval)](https://pypi.org/project/faster-coco-eval)
[![PyPI Downloads](https://img.shields.io/pypi/dm/faster-coco-eval.svg?label=PyPI%20downloads)](https://pypi.org/project/faster-coco-eval/)

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/faster-coco-eval.svg)](https://anaconda.org/conda-forge/faster-coco-eval)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/faster-coco-eval.svg)](https://anaconda.org/conda-forge/faster-coco-eval)

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://github.com/MiXaiLL76/faster_coco_eval/wiki)
[![license](https://img.shields.io/github/license/MiXaiLL76/faster_coco_eval.svg)](https://github.com/MiXaiLL76/faster_coco_eval/blob/main/LICENSE)

[![CI - Test](https://github.com/MiXaiLL76/faster_coco_eval/actions/workflows/unittest.yml/badge.svg)](https://github.com/MiXaiLL76/faster_coco_eval/actions/workflows/unittest.yml)

## Why should you replace pycocotools with **faster-coco-eval**?

| Aspect                         | pycocotools                                                                                 | **faster-coco-eval**                                                                                                                                                                      |
| ------------------------------ | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Support & Development**      | Outdated and not actively maintained. Issues and incompatibilities arise with new releases. | Actively maintained, continuously evolving, and regularly updated with new features and bug fixes.                                                                                        |
| **Transparency & Reliability** | Lacks comprehensive testing, making updates risky and results less predictable.             | Emphasizes extensive test coverage and code quality, ensuring trustworthy and reliable results.                                                                                           |
| **Performance**                | Significantly slower, especially on large datasets or distributed workloads.                | Several times faster due to C++ optimizations and modern algorithms.                                                                                                                      |
| **Functionality**              | Limited to basic COCO format evaluation.                                                    | Offers extended metrics, support for new IoU types, compatibility with more datasets (e.g., CrowdPose, LVIS), advanced visualizations, and seamless integration with PyTorch/TorchVision. |

______________________________________________________________________

**By choosing faster-coco-eval, you benefit from:**

- Reliability and confidence in your results
- High processing speed
- Modern functionality and support for new tasks
- An active community and prompt response to your requests

Switch to **faster-coco-eval** and experience a new standard in working with COCO annotations!

## Install

### Basic implementation identical to pycocotools

```bash
pip install faster-coco-eval
```

### Additional visualization options

> Only 1 additional package needed opencv-python-headless

```bash
pip install faster-coco-eval[extra]
```

### Conda install

```bash
conda install conda-forge::faster-coco-eval
```

### Basic usage

```py
import faster_coco_eval

# Replace pycocotools with faster_coco_eval
faster_coco_eval.init_as_pycocotools()

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

anno = COCO(str(anno_json))  # init annotations api
pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)

val = COCOeval(anno, pred, "bbox")
val.evaluate()
val.accumulate()
val.summarize()

```

## Faster-COCO-Eval base

This package wraps a facebook C++ implementation of COCO-eval operations found in the
[pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) package.
This implementation greatly speeds up the evaluation time
for coco's AP metrics, especially when dealing with a high number of instances in an image.

## Comparison

For our use case with a test dataset of 5000 images from the coco val dataset.
Testing was carried out using the mmdetection framework and the eval_metric.py script. The indicators are presented below.

Visualization of testing **colab_example.ipynb** available in directory [examples/comparison](https://nbviewer.org/github/MiXaiLL76/faster_coco_eval/blob/main/examples/comparison)

- [mmdet example](https://nbviewer.org/github/MiXaiLL76/faster_coco_eval/blob/main/examples/comparison/mmdet/colab_example.ipynb)
- [ultralytics example](https://nbviewer.org/github/MiXaiLL76/faster_coco_eval/blob/main/examples/comparison/ultralytics/colab_example.ipynb)

### Summary for 5000 imgs

| Type | faster-coco-eval | pycocotools | Profit |
| :--- | ---------------: | ----------: | -----: |
| bbox |            5.812 |       22.72 |  3.909 |
| segm |            7.413 |      24.434 |  3.296 |

## Feautures

This library provides not only validation functions, but also error visualization functions. Including visualization of errors in the image.
You can study in more detail in the [examples](https://mixaill76.github.io/faster_coco_eval/examples.html) and [Wiki](https://github.com/MiXaiLL76/faster_coco_eval/wiki).

## Usage

Code examples for using the library are available on the [Wiki](https://github.com/MiXaiLL76/faster_coco_eval/wiki)

### Examples

- [Eval example](https://mixaill76.github.io/faster_coco_eval/examples/eval_example.html)
- [Curve example](https://mixaill76.github.io/faster_coco_eval/examples/curve_example.html)
- [LVIS example](https://mixaill76.github.io/faster_coco_eval/examples/lvis_example.html)
- [Crowdpose example](https://mixaill76.github.io/faster_coco_eval/examples/crowdpose_example.html)
- [Custom keypoints example](https://mixaill76.github.io/faster_coco_eval/examples/ced_example.html)
- [showAnns example](https://mixaill76.github.io/faster_coco_eval/examples/show_example.html)

## Update history

Available via link [history.md](https://mixaill76.github.io/faster_coco_eval/history.html)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MiXaiLL76/faster_coco_eval&type=Date)](https://star-history.com/#MiXaiLL76/faster_coco_eval&Date)

## License

The original module was licensed with apache 2, I will continue with the same license.
Distributed under the apache version 2.0 license, see [license](https://github.com/MiXaiLL76/faster_coco_eval/blob/main/LICENSE) for more information.

## Citation

If you use this benchmark in your research, please cite this project.

```
@article{faster-coco-eval,
  title   = {{Faster-COCO-Eval}: Faster interpretation of the original COCOEval},
  author  = {MiXaiLL76},
  year    = {2024}
}
```
