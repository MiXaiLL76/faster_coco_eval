#!/usr/bin/python3

import os
import unittest

from faster_coco_eval import COCO, COCOeval_faster


class TestBaseLvis(unittest.TestCase):
    """Test basic LVIS functionality."""

    prepared_coco_in_dict = None
    prepared_anns = None

    def setUp(self):
        gt_file = "lvis_dataset/lvis_val_100.json"
        dt_file = "lvis_dataset/lvis_results_100.json"

        if not os.path.exists(gt_file):
            gt_file = os.path.join("tests", gt_file)
            dt_file = os.path.join("tests", dt_file)

        self.prepared_coco_in_dict = COCO.load_json(gt_file)
        self.prepared_anns = COCO.load_json(dt_file)

    def test_coco_eval(self):
        stats_as_dict = {
            "AP_all": 0.3676645003471999,
            "AP_50": 0.626197183778713,
            "AP_75": 0.3842680457694463,
            "AP_small": 0.30144006848393434,
            "AP_medium": 0.4383116520502349,
            "AP_large": 0.44698568700634994,
            "AR_all": 0.4298204126791178,
            "AR_second": 0.0,
            "AR_third": 0.0,
            "AR_small": 0.3451549077565635,
            "AR_medium": 0.48783590386221964,
            "AR_large": 0.5153266620657926,
            "AR_50": 0.7196670897103968,
            "AR_75": 0.44532527811852934,
            "APr": 0.0,
            "APc": 0.2743466491059044,
            "APf": 0.3875839974389359,
        }

        iouType = "bbox"
        cocoGt = COCO(self.prepared_coco_in_dict)
        cocoDt = cocoGt.loadRes(self.prepared_anns)

        cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType, lvis_style=True)
        cocoEval.params.maxDets = [300]

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        self.assertEqual(cocoEval.stats_as_dict, stats_as_dict)


if __name__ == "__main__":
    unittest.main()
