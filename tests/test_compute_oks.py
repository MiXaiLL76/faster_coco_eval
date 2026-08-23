"""Regression coverage for vectorized Object Keypoint Similarity
computation."""

import numpy as np

from faster_coco_eval import COCO
from faster_coco_eval.core.cocoeval import COCOeval


def test_compute_oks_preserves_visible_and_unlabeled_ground_truth_rows():
    """Keep fixed OKS values for visible and unlabeled ground-truth
    keypoints."""
    ground_truth = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0.0, 0.0, 30.0, 30.0],
                "area": 900.0,
                "iscrowd": 0,
                "keypoints": [10.0, 10.0, 2, 20.0, 20.0, 1, 0.0, 0.0, 0],
                "num_keypoints": 2,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [20.0, 20.0, 10.0, 10.0],
                "area": 100.0,
                "iscrowd": 0,
                "keypoints": [0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0],
                "num_keypoints": 0,
            },
        ],
    }
    detections = [
        {
            "image_id": 1,
            "category_id": 1,
            "score": 0.9,
            "keypoints": [11.0, 9.0, 2, 18.0, 22.0, 2, 30.0, 30.0, 2],
        },
        {
            "image_id": 1,
            "category_id": 1,
            "score": 0.8,
            "keypoints": [0.0, 0.0, 2, 0.0, 0.0, 2, 0.0, 0.0, 2],
        },
        {
            "image_id": 1,
            "category_id": 1,
            "score": 0.7,
            "keypoints": [40.0, 40.0, 2, 40.0, 40.0, 2, 40.0, 40.0, 2],
        },
    ]
    coco_gt = COCO(ground_truth)
    evaluator = COCOeval(coco_gt, coco_gt.loadRes(detections), "keypoints", kpt_oks_sigmas=[0.1, 0.1, 0.1])
    evaluator.params.maxDets = [3]
    evaluator._prepare()

    expected = np.array([
        [0.9337218969653591, 0.9608323008615317],
        [0.031095734680320564, 1.388794386496407e-11],
        [7.4726762063626715e-06, 1.0],
    ])
    np.testing.assert_array_equal(evaluator.computeOks(1, 1), expected)
