"""Regression coverage for C++ input-validation boundaries."""

import math

import faster_coco_eval.mask_api_new_cpp as _mask
import numpy as np
import pytest

from faster_coco_eval import COCO, COCOeval_faster


def test_to_bbox_returns_zero_box_for_empty_foreground_rle():
    """An RLE with only a background run has an empty bounding box."""
    rle = _mask.frUncompressedRLE([{"size": [2, 2], "counts": [4]}])
    bbox = _mask.toBbox(rle)

    np.testing.assert_array_equal(bbox, [[0.0, 0.0, 0.0, 0.0]])


def test_rle_varint_rejects_more_than_thirteen_chunks():
    """Malformed compressed RLE cannot shift a varint past int64 width."""
    with pytest.raises(RuntimeError, match="varint exceeds 64-bit width"):
        _mask.RLE("P" * 13 + "0", 1, 1)


def test_evaluator_rejects_nan_detection_score_before_sorting():
    """NaN scores cannot enter the native strict-weak-order comparator."""
    coco_gt = COCO({
        "images": [{"id": 1, "height": 10, "width": 10}],
        "categories": [{"id": 1, "name": "object"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 2, 2],
                "area": 4,
                "iscrowd": 0,
            }
        ],
    })
    coco_dt = coco_gt.loadRes([{"image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2], "score": math.nan}])

    evaluator = COCOeval_faster(coco_gt, coco_dt, iouType="bbox")
    with pytest.raises(ValueError, match="Detection scores must not be NaN"):
        evaluator.evaluate()
