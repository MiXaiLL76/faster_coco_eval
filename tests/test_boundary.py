#!/usr/bin/python3

import os
import unittest

import numpy as np

import faster_coco_eval.core.mask as mask_util
from faster_coco_eval import COCO, COCOeval_faster


class TestBoundary(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        # fmt: off
        segm = [
            [0, 0, 15, 20, 20, 10, 20, 30, 20, 10, 10,
             10, 50, 50, 70, 60, 60, 60, 40, 50, 10, 60, 0, 0],
            [50, 20, 70, 20, 70, 40, 50, 20],
        ]

        mini_mask = np.array([
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0]],
            dtype=np.uint8,
        )

        self.mini_mask_boundry = np.array([
            [[0], [0], [1], [1], [1]],
            [[0], [1], [1], [1], [0]],
            [[0], [1], [0], [1], [0]],
            [[0], [1], [1], [1], [1]],
            [[0], [0], [0], [1], [0]]],
            dtype=np.uint8,
        )
        # fmt: on
        self.mini_mask_rle = mask_util.encode(mini_mask)
        self.rle_80_70 = mask_util.segmToRle(segm, 80, 70)

        self.gt_file = os.path.join("dataset", "gt_dataset.json")
        self.dt_file = os.path.join("dataset", "dt_dataset.json")

        if not os.path.exists(self.gt_file):
            self.gt_file = os.path.join(os.path.dirname(__file__), self.gt_file)
            self.dt_file = os.path.join(os.path.dirname(__file__), self.dt_file)

        prepared_anns = COCO.load_json(self.dt_file)

        cocoGt = COCO(self.gt_file)
        self.prepared_anns_droped_bbox = [
            {
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "iscrowd": ann["iscrowd"],
                "id": ann["id"],
                "score": ann["score"],
                "segmentation": mask_util.merge(
                    mask_util.frPyObjects(
                        ann["segmentation"],
                        cocoGt.imgs[ann["image_id"]]["height"],
                        cocoGt.imgs[ann["image_id"]]["width"],
                    )
                ),
            }
            for ann in prepared_anns
        ]

    def test_rleToBoundary_all(self):
        if not mask_util.opencv_available:
            raise unittest.SkipTest("OpenCV is not available. Skipping test.")

        mask_api_rle = mask_util.rleToBoundary(self.rle_80_70, backend="mask_api")
        opencv_rle = mask_util.rleToBoundary(self.rle_80_70, backend="opencv")
        self.assertDictEqual(mask_api_rle, opencv_rle)

        mask_api_rle_mask = mask_util.decode([mask_util.rleToBoundary(self.mini_mask_rle, backend="mask_api")])
        opencv_rle_mask = mask_util.decode([mask_util.rleToBoundary(self.mini_mask_rle, backend="opencv")])

        self.assertTrue(np.array_equal(mask_api_rle_mask, self.mini_mask_boundry))
        self.assertTrue(np.array_equal(opencv_rle_mask, self.mini_mask_boundry))

    def test_rleToBoundary_mask_api(self):
        mask_api_rle_mask = mask_util.decode([mask_util.rleToBoundary(self.mini_mask_rle, backend="mask_api")])
        self.assertTrue(np.array_equal(mask_api_rle_mask, self.mini_mask_boundry))

    def test_rleToBoundary_opencv(self):
        if not mask_util.opencv_available:
            raise unittest.SkipTest("OpenCV is not available. Skipping test.")

        opencv_rle_mask = mask_util.decode([mask_util.rleToBoundary(self.mini_mask_rle, backend="opencv")])
        self.assertTrue(np.array_equal(opencv_rle_mask, self.mini_mask_boundry))

    def test_boundary_eval(self):
        stats_as_dict = {
            # the following values (except for mIoU and mAUC_50) have been
            # obtained by running the original boundary_iou_api on
            # gt_dataset and dt_dataset
            "AP_all": 0.6947194719471946,
            "AP_50": 0.6947194719471946,
            "AP_75": 0.6947194719471946,
            "AP_small": -1.0,
            "AP_medium": 0.7367986798679867,
            "AP_large": 0.0,
            "AR_all": 0.6666666666666666,
            "AR_second": 0.75,
            "AR_third": 0.75,
            "AR_small": -1.0,
            "AR_medium": 0.7916666666666666,
            "AR_large": 0.0,
            "AR_50": 0.75,
            "AR_75": 0.75,
        }

        cocoGt = COCO(self.gt_file)
        cocoDt = cocoGt.loadRes(self.prepared_anns_droped_bbox)
        cocoEval = COCOeval_faster(cocoGt, cocoDt, "boundary")

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        self.assertAlmostEqual(cocoEval.stats_as_dict, stats_as_dict, places=10)


if __name__ == "__main__":
    unittest.main()
