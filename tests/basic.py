#!/usr/bin/python3

import json
import unittest
from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.extra import PreviewResults


def load(file):
    with open(file) as io:
        _data = json.load(io)
    return _data


class TestBaseCoco(unittest.TestCase):
    def test_coco(self):
        prepared_coco_in_dict = load("dataset/gt_dataset.json")
        prepared_anns = load("dataset/dt_dataset.json")

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

        cocoGt = COCO(prepared_coco_in_dict)
        cocoDt = cocoGt.loadRes(prepared_anns)

        cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType, extra_calc=True)
        cocoEval.params.maxDets = [len(cocoGt.anns)]

        if not useCats:
            cocoEval.params.useCats = 0  # Выключение labels

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        self.assertEqual(cocoEval.stats_as_dict, stats_as_dict)


class TestConfusionMatrix(unittest.TestCase):
    def test_coco(self):
        prepared_coco_in_dict = load("dataset/gt_dataset.json")
        prepared_anns = load("dataset/dt_dataset.json")

        prepared_result = [
            [2.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        ]

        iouType = "segm"
        useCats = False

        cocoGt = COCO(prepared_coco_in_dict)
        cocoDt = cocoGt.loadRes(prepared_anns)

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
