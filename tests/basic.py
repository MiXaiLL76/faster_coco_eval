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
        prepared_coco_in_dict = load('data/gt_cat_dog.json')
        prepared_anns = load('data/dt_cat_dog.json')
        stats_as_dict = {
            'AP_all': 0.6084394153701084,
            'AP_50': 0.7383309759547382,
            'AP_75': 0.7383309759547382,
            'AP_small': -1.0,
            'AP_medium': -1.0,
            'AP_large': 0.6084394153701084,
            'AR_all': 0.7166666666666666,
            'AR_second': 0.0,
            'AR_third': 0.0,
            'AR_small': -1.0,
            'AR_medium': -1.0,
            'AR_large': 0.7166666666666666,
            'AR_50': 0.8333333333333334,
            'AR_75': 0.8333333333333334,
            'mIoU': 0.9042780340786216,
            'mAUC_50': 0.7357142857142857,
        }

        iouType = 'segm'
        useCats = False

        cocoGt = COCO(prepared_coco_in_dict)
        cocoDt = cocoGt.loadRes(prepared_anns)

        cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType)
        cocoEval.params.maxDets = [len(cocoGt.anns)]

        if not useCats:
            cocoEval.params.useCats = 0  # Выключение labels

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        self.assertEqual(cocoEval.stats_as_dict, stats_as_dict)


class TestConfusionMatrix(unittest.TestCase):
    def test_coco(self):
        prepared_coco_in_dict = load('data/gt_cat_dog.json')
        prepared_anns = load('data/dt_cat_dog.json')
        prepared_result = [[2, 0, 2, 1], [0, 3, 0, 0]]

        iouType = 'segm'
        useCats = False

        cocoGt = COCO(prepared_coco_in_dict)
        cocoDt = cocoGt.loadRes(prepared_anns)

        results = PreviewResults(
            cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType, iou_tresh=0.5, useCats=useCats)
        result_cm = results.compute_confusion_matrix().tolist()

        self.assertEqual(result_cm, prepared_result)


if __name__ == '__main__':
    unittest.main()
