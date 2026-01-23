# Faster-COCO-Eval

**The Fastest, Most Reliable COCO Evaluation Library for Computer Vision**

[![PyPI](https://img.shields.io/pypi/v/faster-coco-eval)](https://pypi.org/project/faster-coco-eval)
[![PyPI Downloads](https://img.shields.io/pypi/dm/faster-coco-eval.svg?label=PyPI%20downloads)](https://pypi.org/project/faster-coco-eval/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/faster-coco-eval.svg)](https://anaconda.org/conda-forge/faster-coco-eval)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/faster-coco-eval.svg)](https://anaconda.org/conda-forge/faster-coco-eval)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://github.com/MiXaiLL76/faster_coco_eval/wiki)
[![license](https://img.shields.io/github/license/MiXaiLL76/faster_coco_eval.svg)](https://github.com/MiXaiLL76/faster_coco_eval/blob/main/LICENSE)
[![CI - Test](https://github.com/MiXaiLL76/faster_coco_eval/actions/workflows/unittest.yml/badge.svg)](https://github.com/MiXaiLL76/faster_coco_eval/actions/workflows/unittest.yml)

## Replace pycocotools with Faster-COCO-Eval Today

| Aspect                         | pycocotools                                                                                 | **faster-coco-eval**                                                                                                                                                                      |
| ------------------------------ | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Support & Development**      | Outdated and not actively maintained. Issues and incompatibilities arise with new releases. | Actively maintained, continuously evolving, and regularly updated with new features and bug fixes.                                                                                        |
| **Transparency & Reliability** | Lacks comprehensive testing, making updates risky and results less predictable.             | Emphasizes extensive test coverage and code quality, ensuring trustworthy and reliable results.                                                                                           |
| **Performance**                | Significantly slower, especially on large datasets or distributed workloads.                | **3-4x faster** due to C++ optimizations and modern algorithms.                                                                                                                           |
| **Functionality**              | Limited to basic COCO format evaluation.                                                    | Offers extended metrics, support for new IoU types, compatibility with more datasets (e.g., CrowdPose, LVIS), advanced visualizations, and seamless integration with PyTorch/TorchVision. |
| **Ease of Use**                | Requires manual installation, often with compilation issues.                                 | Simple `pip install`, no compilation required, and drop-in replacement API.                                                                                                               |
| **Visualization**              | Basic plotting capabilities.                                                                 | Advanced error visualization, annotation display, and comprehensive metric analysis tools.                                                                                               |

---

**Key Benefits of Faster-COCO-Eval:**

✅ **Blazing Fast Performance** - Evaluate large datasets in minutes instead of hours  
✅ **Reliable & Trusted** - Extensive test coverage ensures consistent, reproducible results  
✅ **Modern Features** - Support for latest CV tasks, IoU types, and dataset formats  
✅ **Easy to Use** - Drop-in replacement for pycocotools with enhanced API  
✅ **Comprehensive Visualization** - Understand your model's performance with beautiful, informative plots  

**Join thousands of computer vision researchers and engineers who have already switched to Faster-COCO-Eval!**

## Quick Installation

### Option 1: Basic (Drop-in Replacement)

Get started in seconds with the core evaluation functionality:

```bash
pip install faster-coco-eval
```

### Option 2: Full Installation (with Visualization)

For complete functionality including advanced visualization tools:

```bash
pip install faster-coco-eval[extra]
```

### Option 3: Conda Installation

If you use Anaconda/Miniconda:

```bash
conda install conda-forge::faster-coco-eval
```

## 🚀 Quick Start: Drop-in Replacement

Replace pycocotools with Faster-COCO-Eval in **2 lines of code**:

```python
import faster_coco_eval

# This single line replaces pycocotools with faster-coco-eval
faster_coco_eval.init_as_pycocotools()

# Now use the familiar pycocotools API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load annotations and predictions
anno = COCO(str(anno_json))  # Annotations file
pred = anno.loadRes(str(pred_json))  # Predictions file

# Evaluate bounding boxes
val = COCOeval(anno, pred, "bbox")
val.evaluate()
val.accumulate()
val.summarize()

# Or evaluate segmentation masks
val = COCOeval(anno, pred, "segm")
val.evaluate()
val.accumulate()
val.summarize()
```

**That's it! Your existing code will run 3-4x faster with no changes.**

## ⚡ Blazing Fast Performance

Faster-COCO-Eval is built on top of a highly optimized C++ implementation, providing **3-4x faster evaluation** than the standard pycocotools.

### Real-World Performance Benchmark

Tested on 5000 images from the COCO validation dataset using mmdetection framework:

| Evaluation Type | Faster-COCO-Eval (sec) | pycocotools (sec) | Speedup |
|-----------------|------------------------|-------------------|---------|
| Bounding Boxes  | 5.812                  | 22.72             | **3.9x** |
| Segmentation    | 7.413                  | 24.434            | **3.3x** |

**For large datasets, this means hours saved on evaluation time!**

### Colab Examples

See the performance in action:

- [mmdetection comparison](https://nbviewer.org/github/MiXaiLL76/faster_coco_eval/blob/main/examples/comparison/mmdet/colab_example.ipynb)
- [ultralytics comparison](https://nbviewer.org/github/MiXaiLL76/faster_coco_eval/blob/main/examples/comparison/ultralytics/colab_example.ipynb)

## 🎯 Powerful Features

Faster-COCO-Eval goes beyond basic evaluation with these advanced capabilities:

### Core Evaluation
- **Drop-in pycocotools replacement** - No code changes needed
- **Support for all COCO metric types**: bbox, segm, keypoints
- **LVIS (Large Vocabulary Instance Segmentation) evaluation**
- **CrowdPose and custom keypoint datasets**
- **Multiple IoU types**: standard, rotated, and custom IoU definitions

### Advanced Visualization
- **Error visualization**: See where your model is making mistakes
- **Annotation display**: Visualize ground truth and predictions together
- **Metric curves**: Precision-recall curves, class-wise performance
- **Confusion matrices and error analysis**
- **Interactive Jupyter notebook examples**

### Modern Integrations
- **PyTorch/TorchVision compatibility**
- **Seamless integration with mmdetection, Detectron2, and YOLO frameworks**
- **Distributed evaluation support**
- **Memory optimized for large datasets**

### Additional Tools
- **Boundary evaluation for segmentation tasks**
- **Custom dataset support**
- **Comprehensive API documentation**
- **Extensive test coverage and reliability**

## 📚 Comprehensive Documentation

### Usage Examples

Explore practical, runnable examples in Jupyter notebooks:

- [Basic Evaluation](https://mixaill76.github.io/faster_coco_eval/examples/eval_example.html) - Get started with COCO evaluation
- [Metric Curves](https://mixaill76.github.io/faster_coco_eval/examples/curve_example.html) - Precision-recall and metric visualization
- [LVIS Evaluation](https://mixaill76.github.io/faster_coco_eval/examples/lvis_example.html) - Large vocabulary instance segmentation
- [CrowdPose Evaluation](https://mixaill76.github.io/faster_coco_eval/examples/crowdpose_example.html) - Keypoint detection for crowded scenes
- [Custom Keypoints](https://mixaill76.github.io/faster_coco_eval/examples/ced_example.html) - Extend to custom keypoint datasets
- [Annotation Visualization](https://mixaill76.github.io/faster_coco_eval/examples/show_example.html) - Display and analyze annotations

### Detailed Documentation

- [Official Wiki](https://github.com/MiXaiLL76/faster_coco_eval/wiki) - Complete API reference and guides
- [Changelog](https://mixaill76.github.io/faster_coco_eval/history.html) - Latest updates and improvements
- [API Documentation](https://github.com/MiXaiLL76/faster_coco_eval/wiki) - Detailed function documentation

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MiXaiLL76/faster_coco_eval&type=Date)](https://star-history.com/#MiXaiLL76/faster_coco_eval&Date)

## 📄 License

Faster-COCO-Eval is distributed under the Apache 2.0 license. See [LICENSE](https://github.com/MiXaiLL76/faster_coco_eval/blob/main/LICENSE) for more information.

## 📚 Citation

If you use Faster-COCO-Eval in your research, please cite:

```bibtex
@article{faster-coco-eval,
  title   = {{Faster-COCO-Eval}: Faster and Enhanced COCO Evaluation Library},
  author  = {MiXaiLL76},
  year    = {2024}
}
```

## 🤝 Contributing

We welcome contributions! Check out our [CONTRIBUTING.md](https://github.com/MiXaiLL76/faster_coco_eval/blob/main/CONTRIBUTING.md) for guidelines on how to get started.

## 🐛 Issues and Support

If you encounter any issues or have questions:

1. Check the [Wiki](https://github.com/MiXaiLL76/faster_coco_eval/wiki) for common solutions
2. Search existing [issues](https://github.com/MiXaiLL76/faster_coco_eval/issues)
3. Open a new issue with detailed information about your problem

## 🚀 Get Started Today

```bash
pip install faster-coco-eval[extra]
```

**Replace pycocotools with Faster-COCO-Eval and experience evaluation at lightning speed!**
