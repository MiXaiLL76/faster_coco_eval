import sys
import unittest
from unittest import TestCase

import faster_coco_eval
import faster_coco_eval.core


class TestInitAsPycocotools(TestCase):
    def setUp(self):
        faster_coco_eval.init_as_pycocotools()

    def tearDown(self):
        del sys.modules["pycocotools"]
        del sys.modules["pycocotools.coco"]
        del sys.modules["pycocotools.cocoeval"]
        del sys.modules["pycocotools.mask"]

    def test_evaluate(self):
        self.assertEqual(sys.modules["pycocotools"], faster_coco_eval)
        self.assertEqual(sys.modules["pycocotools.coco"], faster_coco_eval.core.coco)
        self.assertEqual(
            sys.modules["pycocotools.cocoeval"],
            faster_coco_eval.core.faster_eval_api,
        )
        self.assertEqual(sys.modules["pycocotools.mask"], faster_coco_eval.core.mask)


if __name__ == "__main__":
    unittest.main()
