import json
import os.path as osp
import tempfile
import unittest
from unittest import TestCase

import numpy as np
from parameterized import parameterized
from pycocotools.coco import COCO as origCOCO
from pycocotools.cocoeval import COCOeval as origCOCOeval

import faster_coco_eval.core.mask as mask_util
from faster_coco_eval import COCO, COCOeval_faster


class TestCocoMetric(TestCase):

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
        }

        with open(json_name, "w") as fd:
            json.dump(fake_json, fd)

    def _create_dummy_results(self):
        bboxes = np.array(
            [
                [50, 60, 70, 80],
                [100, 120, 130, 150],
                [150, 160, 190, 200],
                [250, 260, 350, 360],
            ]
        )
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

        eval_results = {
            key: round(cocoEval.stats[i], 4)
            for i, key in enumerate(list(target))
        }

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
        eval_results = {
            key: round(cocoEval.stats[i], 4)
            for i, key in enumerate(list(target))
        }

        self.assertDictEqual(eval_results, target)


if __name__ == "__main__":
    unittest.main()
