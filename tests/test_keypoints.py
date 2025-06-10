import os
import unittest
from unittest import TestCase

from parameterized import parameterized

try:
    from pycocotools.coco import COCO as origCOCO
    from pycocotools.cocoeval import COCOeval as origCOCOeval
except ImportError:
    origCOCO = None
    origCOCOeval = None

from faster_coco_eval import COCO, COCOeval_faster


class TestKeypointsMetric(TestCase):
    maxDiff = None

    def setUp(self):
        self.gt_file = os.path.join("keypoints_dataset", "gt_dataset.json")
        self.dt_file = os.path.join("keypoints_dataset", "dt_dataset.json")

        if not os.path.exists(self.gt_file):
            self.gt_file = os.path.join(os.path.dirname(__file__), self.gt_file)
            self.dt_file = os.path.join(os.path.dirname(__file__), self.dt_file)

        self.results = [
            0.5048844884488449,
            0.7227722772277227,
            0.6336633663366337,
            0.46633663366336636,
            0.7504950495049505,
            0.5181818181818182,
            0.7272727272727273,
            0.6363636363636364,
            0.4666666666666666,
            0.75,
        ]

    @parameterized.expand([(COCO, COCOeval_faster), (origCOCO, origCOCOeval)])
    def test_evaluate(self, coco_cls, cocoeval_cls):
        if coco_cls is None:
            raise unittest.SkipTest("Skipping pycocotools test.")

        cocoGt = coco_cls(self.gt_file)
        cocoGt.info()

        cocoDt = cocoGt.loadRes(self.dt_file)

        cocoEval = cocoeval_cls(cocoGt, cocoDt, "keypoints")

        cocoEval.evaluate()
        cocoEval.accumulate()

        if cocoeval_cls is COCOeval_faster:
            print(str(cocoEval))
        else:
            cocoEval.summarize()

        self.assertListEqual(cocoEval.stats.tolist(), self.results)

    def test_evaluate_bad_config(self):
        cocoGt = COCO(self.gt_file)
        cocoDt = cocoGt.loadRes(self.dt_file)

        cocoEval = COCOeval_faster(cocoGt, cocoDt, "keypoints", lvis_style=True, print_function=print)
        self.assertFalse(cocoEval.lvis_style)
        self.assertEqual(cocoEval.cocoGt.print_function, print)

        def sub_print(*args, **kwargs):
            print(*args, **kwargs)

        cocoEval.print_function = sub_print
        self.assertEqual(cocoEval.print_function, sub_print)


if __name__ == "__main__":
    unittest.main()
