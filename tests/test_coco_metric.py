import json
import os.path as osp
import tempfile
import unittest
from unittest import TestCase

import numpy as np
from parameterized import parameterized

try:
    from pycocotools.coco import COCO as origCOCO
    from pycocotools.cocoeval import COCOeval as origCOCOeval
except ImportError:
    origCOCO = None
    origCOCOeval = None

import faster_coco_eval.core.mask as mask_util
from faster_coco_eval import COCO, COCOeval_faster


class TestCocoMetric(TestCase):
    maxDiff = None

    def _create_dummy_coco_json(self, json_name):
        dummy_mask = np.zeros((10, 10), order="F", dtype=np.uint8)
        dummy_mask[:5, :5] = 1
        rle_mask = mask_util.encode(dummy_mask)
        rle_mask["counts"] = rle_mask["counts"].decode("utf-8")
        image = {
            "id": 0,
            "width": 640,
            "height": 640,
            "file_name": "fake_name.jpg",
        }

        annotation_1 = {
            "id": 1,
            "image_id": 0,
            "category_id": 0,
            "area": 400,
            "bbox": [50, 60, 20, 20],
            "iscrowd": 0,
            "segmentation": rle_mask,
        }

        annotation_2 = {
            "id": 2,
            "image_id": 0,
            "category_id": 0,
            "area": 900,
            "bbox": [100, 120, 30, 30],
            "iscrowd": 0,
            "segmentation": rle_mask,
        }

        annotation_3 = {
            "id": 3,
            "image_id": 0,
            "category_id": 1,
            "area": 1600,
            "bbox": [150, 160, 40, 40],
            "iscrowd": 0,
            "segmentation": rle_mask,
        }

        annotation_4 = {
            "id": 4,
            "image_id": 0,
            "category_id": 0,
            "area": 10000,
            "bbox": [250, 260, 100, 100],
            "iscrowd": 0,
            "segmentation": rle_mask,
        }

        categories = [
            {
                "id": 0,
                "name": "car",
                "supercategory": "car",
            },
            {
                "id": 1,
                "name": "bicycle",
                "supercategory": "bicycle",
            },
        ]

        fake_json = {
            "images": [image],
            "annotations": [
                annotation_1,
                annotation_2,
                annotation_3,
                annotation_4,
            ],
            "categories": categories,
            "info": "fake info",
        }

        with open(json_name, "w") as fd:
            json.dump(fake_json, fd)

    def _create_dummy_results(self):
        bboxes = np.array([
            [50, 60, 70, 80],
            [100, 120, 130, 150],
            [150, 160, 190, 200],
            [250, 260, 350, 360],
        ])
        scores = np.array([1.0, 0.98, 0.96, 0.95])
        labels = np.array([0, 0, 1, 0])

        annotations = []
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            w, h = x2 - x1, y2 - y1

            dummy_mask = np.zeros((10, 10), order="F", dtype=np.uint8)
            dummy_mask[:5, :5] = 1
            rle_mask = mask_util.encode(dummy_mask)
            rle_mask["counts"] = rle_mask["counts"].decode("utf-8")

            annotation = {
                "id": (i + 1),
                "image_id": 0,
                "category_id": labels[i],
                "area": w * h,
                "bbox": [x1, y1, w, h],
                "score": scores[i],
                "segmentation": rle_mask,
            }
            annotations.append(annotation)
        return annotations

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    @parameterized.expand([(COCO, COCOeval_faster), (origCOCO, origCOCOeval)])
    def test_evaluate(self, coco_cls, cocoeval_cls):
        if coco_cls is None:
            raise unittest.SkipTest("Skipping pycocotools test.")

        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, "fake_data.json")
        self._create_dummy_coco_json(fake_json_file)
        dummy_pred = self._create_dummy_results()

        fake_GT = coco_cls(fake_json_file)
        fake_DT = fake_GT.loadRes(dummy_pred)
        cocoEval = cocoeval_cls(fake_GT, fake_DT, iouType="bbox")

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        target = {
            "coco/bbox_mAP": 1.0,
            "coco/bbox_mAP_50": 1.0,
            "coco/bbox_mAP_75": 1.0,
            "coco/bbox_mAP_s": 1.0,
            "coco/bbox_mAP_m": 1.0,
            "coco/bbox_mAP_l": 1.0,
        }

        eval_results = {key: round(cocoEval.stats[i], 4) for i, key in enumerate(list(target))}

        self.assertDictEqual(eval_results, target)

        fake_GT = coco_cls(fake_json_file)
        fake_DT = fake_GT.loadRes(dummy_pred)
        cocoEval = cocoeval_cls(fake_GT, fake_DT, iouType="segm")

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        target = {
            "coco/segm_mAP": 1.0,
            "coco/segm_mAP_50": 1.0,
            "coco/segm_mAP_75": 1.0,
            "coco/segm_mAP_s": 1.0,
            "coco/segm_mAP_m": 1.0,
            "coco/segm_mAP_l": 1.0,
        }
        eval_results = {key: round(cocoEval.stats[i], 4) for i, key in enumerate(list(target))}

        self.assertDictEqual(eval_results, target)

    @parameterized.expand([(COCO, COCOeval_faster), (origCOCO, origCOCOeval)])
    def test_detectron2_eval(self, coco_cls, cocoeval_cls):
        if coco_cls is None:
            raise unittest.SkipTest("Skipping pycocotools test.")

        # https://github.com/facebookresearch/detectron2/blob/main/tests/data/test_coco_evaluation.py
        # A small set of images/categories from COCO val
        # fmt: off
        detections = [{"image_id": 139, "category_id": 1, "bbox": [417.3332824707031, 159.27003479003906, 47.66064453125, 143.00193786621094], "score": 0.9949821829795837, "segmentation": {"size": [426, 640], "counts": "Tc`52W=3N0N4aNN^E7]:4XE1g:8kDMT;U100000001O1gE[Nk8h1dFiNY9Z1aFkN]9g2J3NdN`FlN`9S1cFRN07]9g1bFoM6;X9c1cFoM=8R9g1bFQN>3U9Y30O01OO1O001N2O1N1O4L4L5UNoE3V:CVF6Q:@YF9l9@ZF<k9[O`F=];HYnX2"}}, {"image_id": 139, "category_id": 1, "bbox": [383.5909118652344, 172.0777587890625, 17.959075927734375, 36.94813537597656], "score": 0.7685421705245972, "segmentation": {"size": [426, 640], "counts": "lZP5m0Z<300O100O100000001O00]OlC0T<OnCOT<OnCNX<JnC2bQT3"}}, {"image_id": 139, "category_id": 1, "bbox": [457.8359069824219, 158.88027954101562, 9.89764404296875, 8.771820068359375], "score": 0.07092753797769547, "segmentation": {"size": [426, 640], "counts": "bSo54T=2N2O1001O006ImiW2"}}] # noqa
        gt_annotations = {"info" : "fake info", "categories": [{"supercategory": "person", "id": 1, "name": "person"}, {"supercategory": "furniture", "id": 65, "name": "bed"}], "images": [{"license": 4, "file_name": "000000000285.jpg", "coco_url": "http://images.cocodataset.org/val2017/000000000285.jpg", "height": 640, "width": 586, "date_captured": "2013-11-18 13:09:47", "flickr_url": "http://farm8.staticflickr.com/7434/9138147604_c6225224b8_z.jpg", "id": 285}, {"license": 2, "file_name": "000000000139.jpg", "coco_url": "http://images.cocodataset.org/val2017/000000000139.jpg", "height": 426, "width": 640, "date_captured": "2013-11-21 01:34:01", "flickr_url": "http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg", "id": 139}], "annotations": [{"segmentation": [[428.19, 219.47, 430.94, 209.57, 430.39, 210.12, 421.32, 216.17, 412.8, 217.27, 413.9, 214.24, 422.42, 211.22, 429.29, 201.6, 430.67, 181.8, 430.12, 175.2, 427.09, 168.06, 426.27, 164.21, 430.94, 159.26, 440.29, 157.61, 446.06, 163.93, 448.53, 168.06, 448.53, 173.01, 449.08, 174.93, 454.03, 185.1, 455.41, 188.4, 458.43, 195.0, 460.08, 210.94, 462.28, 226.61, 460.91, 233.76, 454.31, 234.04, 460.08, 256.85, 462.56, 268.13, 465.58, 290.67, 465.85, 293.14, 463.38, 295.62, 452.66, 295.34, 448.26, 294.52, 443.59, 282.7, 446.06, 235.14, 446.34, 230.19, 438.09, 232.39, 438.09, 221.67, 434.24, 221.12, 427.09, 219.74]], "area": 2913.1103999999987, "iscrowd": 0, "image_id": 139, "bbox": [412.8, 157.61, 53.05, 138.01], "category_id": 1, "id": 230831}, {"segmentation": [[384.98, 206.58, 384.43, 199.98, 385.25, 193.66, 385.25, 190.08, 387.18, 185.13, 387.18, 182.93, 386.08, 181.01, 385.25, 178.81, 385.25, 175.79, 388.0, 172.76, 394.88, 172.21, 398.72, 173.31, 399.27, 176.06, 399.55, 183.48, 397.9, 185.68, 395.15, 188.98, 396.8, 193.38, 398.45, 194.48, 399.0, 205.75, 395.43, 207.95, 388.83, 206.03]], "area": 435.1449499999997, "iscrowd": 0, "image_id": 139, "bbox": [384.43, 172.21, 15.12, 35.74], "category_id": 1, "id": 233201}]} # noqa
        # fmt: on
        fake_gt_annotations_file = osp.join(self.tmp_dir.name, "gt_annotations.json")

        with open(fake_gt_annotations_file, "w") as f:
            json.dump(gt_annotations, f)

        coco_api = coco_cls(fake_gt_annotations_file)
        coco_dt = coco_api.loadRes(detections)
        target = {
            "bbox": [
                0.7504950495049505,
                1.0,
                1.0,
                0.7,
                0.8,
                -1.0,
                0.4,
                0.75,
                0.75,
                0.7,
                0.8,
                -1.0,
            ],
            "segm": [
                0.7252475247524752,
                1.0,
                1.0,
                0.8,
                0.7,
                -1.0,
                0.35,
                0.75,
                0.75,
                0.8,
                0.7,
                -1.0,
            ],
        }
        for iou_type in ["bbox", "segm"]:
            coco_eval = cocoeval_cls(coco_api, coco_dt, iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            for i in range(len(target[iou_type])):
                self.assertAlmostEqual(coco_eval.stats[i], target[iou_type][i], 12)


if __name__ == "__main__":
    unittest.main()
