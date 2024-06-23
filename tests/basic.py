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

        if not os.path.exists(gt_file):
            gt_file = os.path.join("tests", gt_file)
            dt_file = os.path.join("tests", dt_file)

        self.prepared_coco_in_dict = COCO.load_json(gt_file)
        self.prepared_anns = COCO.load_json(dt_file)

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
