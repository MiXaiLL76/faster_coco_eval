# Test Suite Documentation

This directory contains the test suite for `faster_coco_eval`, which validates that the library produces identical results to `pycocotools` while being significantly faster.

## Test Organization

### Core Functionality Tests
- **test_basic.py** - Basic COCO evaluation functionality
- **test_coco_metric.py** - COCO metrics with pycocotools comparison (small examples)
- **test_keypoints.py** - Keypoint evaluation
- **test_cocoapi_fake_data.py** - Tests with synthetic data

### Extensive Comparison Tests
- **test_extensive_pycocotools_comparison.py** - **NEW**: Comprehensive validation against pycocotools with large synthetic datasets

### Dataset-Specific Tests
- **test_lvis_metric.py** - LVIS dataset support
- **test_crowdpose.py** - CrowdPose keypoints dataset

### API and Integration Tests
- **test_init_pycocotools.py** - Drop-in replacement compatibility
- **test_torchmetrics.py** - PyTorch integration (if available)
- **test_mask_api.py** - Mask utilities
- **test_boundary.py** - Boundary evaluation

### Visualization and Utilities
- **test_extra_draw.py**, **test_extra_utils.py**, **test_simple_extra.py** - Visualization features
- **test_ranges.py**, **test_dataset.py** - Utility functions

## Extensive PyCocoTools Comparison Tests

The `test_extensive_pycocotools_comparison.py` module provides comprehensive validation that `faster_coco_eval` produces **identical results** to `pycocotools` across a wide range of scenarios.

### Test Coverage

#### Object Detection (BBox) Tests
Tests bounding box detection with datasets of varying sizes:
- **Small dataset**: 10 images, 5 categories, ~50 annotations
- **Medium dataset**: 50 images, 10 categories, ~500 annotations  
- **Large dataset**: 100 images, 20 categories, ~1500 annotations

Each test validates that both libraries produce identical mAP, mAP@50, mAP@75, and size-specific metrics (small/medium/large objects).

#### Instance Segmentation Tests
Tests segmentation masks with the same dataset size variations as bbox tests. Validates pixel-level mask IoU calculations match exactly between implementations.

#### Keypoint Detection Tests
Tests keypoint pose estimation with datasets containing:
- **Small dataset**: 10 images with 17 keypoints per person
- **Medium dataset**: 50 images with multiple people per image
- **Large dataset**: 100 images with varied keypoint visibility

Validates that OKS (Object Keypoint Similarity) calculations are identical.

#### Edge Cases
- **Perfect predictions**: All predictions match ground truth exactly (IoU=1.0)
- **Low confidence predictions**: Tests with very low-scoring detections
- **Mixed object sizes**: Validates correct assignment to small/medium/large categories

### Test Data Generation

The tests use **synthetic but realistic** COCO-formatted datasets that mimic actual model predictions:

- **Varied image sizes**: Random dimensions between 400x400 and 800x800 pixels
- **Realistic bounding boxes**: Objects categorized as small (<32²), medium (32²-96²), or large (>96²)
- **Segmentation masks**: RLE-encoded binary masks matching bbox regions
- **Keypoint annotations**: 17 keypoints per instance with realistic visibility flags
- **Prediction noise**: Simulated detection errors with bbox jitter and confidence scores
- **False positives**: Includes spurious detections to test precision/recall

### Running the Tests

Run all extensive comparison tests:
```bash
cd tests/
pytest test_extensive_pycocotools_comparison.py -v
```

Run specific test categories:
```bash
# Only bbox tests
pytest test_extensive_pycocotools_comparison.py -k "bbox" -v

# Only segmentation tests
pytest test_extensive_pycocotools_comparison.py -k "segmentation" -v

# Only keypoint tests
pytest test_extensive_pycocotools_comparison.py -k "keypoints" -v

# Only large dataset tests
pytest test_extensive_pycocotools_comparison.py -k "large" -v
```

### Test Success Criteria

Tests pass if and only if:
1. All metrics (mAP, mAP@50, mAP@75, mAP_small, mAP_medium, mAP_large, etc.) are **numerically identical** between `faster_coco_eval` and `pycocotools`
2. Floating-point comparison uses tolerance of `1e-10` (essentially exact)
3. All intermediate calculations (IoU, OKS) produce identical results

### Why These Tests Matter

These extensive tests address the requirement for **confidence in correctness** when using `faster_coco_eval` as a drop-in replacement for `pycocotools`:

- **Broader coverage**: Tests hundreds to thousands of annotations vs. single-digit examples in original tests
- **Real-world scenarios**: Synthetic data mimics actual model predictions with realistic error patterns
- **All task types**: Validates bbox, segmentation, and keypoints independently
- **Edge cases**: Ensures correct behavior in corner cases that might not appear in small datasets
- **Continuous validation**: Runs in CI/CD to catch any regression in numerical accuracy

## Running All Tests

Run the complete test suite:
```bash
cd tests/
pytest --cov=faster_coco_eval .
```

Run tests for a specific Python version (CI/CD runs Python 3.9-3.13):
```bash
pytest --cov=faster_coco_eval . -v
```

## Test Requirements

Install test dependencies:
```bash
pip install "faster-coco-eval[tests]"
```

Or from source:
```bash
cd /path/to/faster_coco_eval
pip install -e ".[tests]"
```

Required packages:
- `pytest` - Test framework
- `pytest-cov` - Coverage reporting
- `parameterized` - Parameterized test cases
- `pycocotools` - Original COCO API for comparison tests
- `numpy` - Numerical operations

## Contributing Tests

When adding new features to `faster_coco_eval`, please:

1. Add corresponding tests that validate **exact equality** with `pycocotools` behavior
2. Use parameterized tests to cover multiple scenarios efficiently
3. Generate synthetic test data programmatically for reproducibility
4. Set `np.random.seed()` for deterministic test data
5. Document what each test validates and why it's important
