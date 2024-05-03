# Faster-COCO-Eval

[![PyPI](https://img.shields.io/pypi/v/faster-coco-eval)](https://pypi.org/project/faster-coco-eval)
[![PyPI Downloads](https://img.shields.io/pypi/dm/faster-coco-eval.svg?label=PyPI%20downloads)](
https://pypi.org/project/faster-coco-eval/)

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://github.com/MiXaiLL76/faster_coco_eval/wiki)
[![license](https://img.shields.io/github/license/MiXaiLL76/faster_coco_eval.svg)](https://github.com/MiXaiLL76/faster_coco_eval/blob/main/LICENSE)

[![CI - Test](https://github.com/MiXaiLL76/faster_coco_eval/actions/workflows/unittest.yml/badge.svg)](https://github.com/MiXaiLL76/faster_coco_eval/actions/workflows/unittest.yml)
<!-- [![Coverage](https://codecov.io/github/MiXaiLL76/faster_coco_eval/coverage.svg?branch=main)](https://codecov.io/gh/MiXaiLL76/faster_coco_eval)       -->

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

Visualization of testing **colab_example.ipynb** available in directory [examples/comparison](https://github.com/MiXaiLL76/faster_coco_eval/blob/main/examples/comparison/mmdet/colab_example.ipynb)
[colab_example.ipynb in google collab](https://colab.research.google.com/drive/1qj392oIU8fmeyIFHCtxCrLA8PQQAMoIs)
Tested with rtmdet model bbox + segm

### Summary for 5000 imgs

| Type | faster-coco-eval | pycocotools | Profit |
| :--- | ---------------: | ----------: | -----: |
| bbox |            5.812 |       22.72 |  3.909 |
| segm |            7.413 |      24.434 |  3.296 |

## Feautures

This library provides not only validation functions, but also error visualization functions. Including visualization of errors in the image.  
You can study in more detail in the [examples](https://github.com/MiXaiLL76/faster_coco_eval/blob/main/examples) and [Wiki](https://github.com/MiXaiLL76/faster_coco_eval/wiki).

## Usage

Code examples for using the library are available on the [Wiki](https://github.com/MiXaiLL76/faster_coco_eval/wiki)

## Update history

Available via link [history.md](https://github.com/MiXaiLL76/faster_coco_eval/blob/main/history.md)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MiXaiLL76/faster_coco_eval&type=Date)](https://star-history.com/#MiXaiLL76/faster_coco_eval&Date)

## License

The original module was licensed with apache 2, I will continue with the same license.
Distributed under the apache version 2.0 license, see [license](LICENSE) for more information.

## Citation

If you use this benchmark in your research, please cite this project.

```
@article{faster-coco-eval,
  title   = {{Faster-COCO-Eval}: Faster interpretation of the original COCOEval},
  author  = {MiXaiLL76},
  year    = {2024}
}
```