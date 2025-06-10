#!/usr/bin/python3
import os
import unittest

import numpy as np

from faster_coco_eval import COCO, COCOeval_faster


class TestCrowdpose(unittest.TestCase):
    """Test Crowdpose functionality."""

    maxDiff = None

    def setUp(self):
        self.gt_file = os.path.join("dataset", "example_crowdpose_val.json")
        self.dt_file = os.path.join("dataset", "example_crowdpose_preds.json")
        self.sigmas = (
            np.array([0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89, 0.79, 0.79]) / 10.0
        )

        if not os.path.exists(self.gt_file):
            self.gt_file = os.path.join(os.path.dirname(__file__), self.gt_file)
            self.dt_file = os.path.join(os.path.dirname(__file__), self.dt_file)

    def test_crowdpose_eval(self):
        stats_as_dict = {
            "AP_all": 0.7877215935879303,
            "AP_50": 0.9881188118811886,
            "AP_75": 0.7314356435643564,
            "AR_all": 0.8222222222222223,
            "AR_50": 1.0,
            "AR_75": 0.7777777777777778,
            "AP_easy": 1.0,
            "AP_medium": 0.9802,
            "AP_hard": 0.4116,
        }

        cocoGt = COCO(self.gt_file)

        cocoDt = cocoGt.loadRes(self.dt_file)

        cocoEval = COCOeval_faster(
            cocoGt, cocoDt, iouType="keypoints_crowd", kpt_oks_sigmas=self.sigmas, use_area=False
        )

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        self.assertAlmostEqual(cocoEval.stats_as_dict, stats_as_dict, places=10)


if __name__ == "__main__":
    unittest.main()
