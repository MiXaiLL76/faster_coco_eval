#!/usr/bin/python3

import json
import unittest
from faster_coco_eval import COCO, COCOeval_faster

def load(file):
    with open(file) as io:
        _data = json.load(io)
    return _data

class TestStringMethods(unittest.TestCase):
    def test_coco(self):
        prepared_coco_in_dict = load('data/eval_all_coco.json')
        prepared_anns         = load('data/result_annotations.json')
        stats_as_dict         = load('data/stats_as_dict.json')

        iouType = 'segm'
        useCats = False

        cocoGt = COCO(prepared_coco_in_dict)
        cocoDt = cocoGt.loadRes(prepared_anns)

        cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType)
        cocoEval.params.maxDets = [len(cocoGt.anns)]

        cocoEval.params.iouThr   = [0.5, 0.75]
        if not useCats:
            cocoEval.params.useCats = 0 # Выключение labels

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        self.assertEqual(cocoEval.stats_as_dict, stats_as_dict)

if __name__ == '__main__':
    unittest.main()