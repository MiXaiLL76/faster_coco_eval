#!/usr/bin/python3

import os
import unittest

from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.extra import PreviewResults


class TestBaseCoco(unittest.TestCase):
    """Test basic COCO functionality."""

    prepared_coco_in_dict = None
    prepared_anns = None

    def setUp(self):
        gt_file = "dataset/gt_dataset.json"
        dt_file = "dataset/dt_dataset.json"
        gt_ignore_test_file = "dataset/gt_ignore_test.json"
        dt_ignore_test_file = "dataset/dt_ignore_test.json"

        if not os.path.exists(gt_file):
            gt_file = os.path.join("tests", gt_file)
            dt_file = os.path.join("tests", dt_file)
            gt_ignore_test_file = os.path.join("tests", gt_ignore_test_file)
            dt_ignore_test_file = os.path.join("tests", dt_ignore_test_file)

        self.prepared_coco_in_dict = COCO.load_json(gt_file)
        self.prepared_anns = COCO.load_json(dt_file)

        self.ignore_coco_in_dict = COCO.load_json(gt_ignore_test_file)
        self.ignore_prepared_anns = COCO.load_json(dt_ignore_test_file)

    def test_ignore_coco_eval(self):
        stats_as_dict = {
            "AP_all": 0.7099009900990099,
            "AP_50": 1.0,
            "AP_75": 0.8415841584158416,
            "AP_small": -1.0,
            "AP_medium": 0.7099009900990099,
            "AP_large": -1.0,
            "AR_1": 0.0,
            "AR_10": 0.45384615384615384,
            "AR_100": 0.7153846153846154,
            "AR_small": -1.0,
            "AR_medium": 0.7153846153846154,
            "AR_large": -1.0,
            "AR_50": 1.0,
            "AR_75": 0.8461538461538461,
        }

        iouType = "bbox"

        cocoGt = COCO(self.ignore_coco_in_dict)
        cocoDt = cocoGt.loadRes(self.ignore_prepared_anns)

        cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType)

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        self.assertEqual(cocoEval.stats_as_dict, stats_as_dict)

    def test_coco_eval(self):
        stats_as_dict = {
            "AP_all": 0.7832783278327835,
            "AP_50": 0.7832783278327836,
            "AP_75": 0.7832783278327836,
            "AP_small": -1.0,
            "AP_medium": 1.0,
            "AP_large": 0.0,
            "AR_all": 0.888888888888889,
            "AR_second": 0.0,
            "AR_third": 0.0,
            "AR_small": -1.0,
            "AR_medium": 1.0,
            "AR_large": 0.0,
            "AR_50": 0.8888888888888888,
            "AR_75": 0.8888888888888888,
            "mIoU": 1.0,
            "mAUC_50": 0.594074074074074,
        }

        iouType = "segm"
        useCats = False

        cocoGt = COCO(self.prepared_coco_in_dict)
        cocoDt = cocoGt.loadRes(self.prepared_anns)

        cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType, extra_calc=True)
        cocoEval.params.maxDets = [len(cocoGt.anns)]

        if not useCats:
            cocoEval.params.useCats = 0  # Выключение labels

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        self.assertEqual(cocoEval.matched, True)
        self.assertEqual(cocoEval.stats_as_dict, stats_as_dict)

    def test_confusion_matrix(self):
        prepared_result = [
            [2.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        ]

        iouType = "segm"
        useCats = False

        cocoGt = COCO(self.prepared_coco_in_dict)
        cocoDt = cocoGt.loadRes(self.prepared_anns)

        results = PreviewResults(
            cocoGt=cocoGt,
            cocoDt=cocoDt,
            iouType=iouType,
            iou_tresh=0.5,
            useCats=useCats,
        )
        result_cm = results.compute_confusion_matrix().tolist()

        self.assertEqual(result_cm, prepared_result)


if __name__ == "__main__":
    unittest.main()
