#!/usr/bin/python3

import os
import unittest

import numpy as np

from faster_coco_eval import COCO, COCOeval_faster


class TestBaseLvis(unittest.TestCase):
    """Test basic LVIS functionality."""

    prepared_coco_in_dict = None
    prepared_anns = None
    maxDiff = None

    def setUp(self):
        self.gt_file = os.path.join("lvis_dataset", "lvis_val_100.json")
        self.dt_file = os.path.join("lvis_dataset", "lvis_results_100.json")

        if not os.path.exists(self.gt_file):
            self.gt_file = os.path.join(os.path.dirname(__file__), self.gt_file)
            self.dt_file = os.path.join(os.path.dirname(__file__), self.dt_file)

        self.prepared_coco_in_dict = COCO.load_json(self.gt_file)
        self.prepared_anns = COCO.load_json(self.dt_file)

        self.prepared_anns_numpy = []
        for ann in self.prepared_anns:
            self.prepared_anns_numpy.append([ann["image_id"]] + ann["bbox"] + [ann["score"]] + [ann["category_id"]])

        self.prepared_anns_numpy = np.array(self.prepared_anns_numpy)

        self.stats_as_dict_result = {
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

    def test_lvis_eval(self):
        iouType = "bbox"
        cocoGt = COCO(self.prepared_coco_in_dict)
        cocoDt = cocoGt.loadRes(self.prepared_anns)

        cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType, lvis_style=True)
        cocoEval.params.maxDets = [300]

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        for key in self.stats_as_dict_result.keys():
            self.assertAlmostEqual(cocoEval.stats_as_dict[key], self.stats_as_dict_result[key], places=10, msg=key)

    def test_loadNumpyAnnotations(self):
        iouType = "bbox"
        cocoGt = COCO(self.prepared_coco_in_dict)
        cocoDt = cocoGt.loadRes(self.prepared_anns_numpy)

        cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType, lvis_style=True)
        cocoEval.params.maxDets = [300]

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        for key in self.stats_as_dict_result.keys():
            self.assertAlmostEqual(cocoEval.stats_as_dict[key], self.stats_as_dict_result[key], places=10, msg=key)

    def test_getAnnIds(self):
        cocoGt = COCO(self.gt_file)
        self.assertEqual(cocoGt.getAnnIds(), list(cocoGt.anns.keys()))

        category_id = 1
        self.assertEqual(
            cocoGt.getAnnIds(catIds=[category_id]),
            [key for key, val in cocoGt.anns.items() if int(val["category_id"]) == category_id],
        )

        image_id = 521509
        category_id = 191
        self.assertEqual(
            cocoGt.getAnnIds(catIds=[category_id], imgIds=[image_id]),
            [
                key
                for key, val in cocoGt.anns.items()
                if (int(val["category_id"]) == category_id) and (int(val["image_id"]) == image_id)
            ],
        )

        image_id = 521509
        areaRng = [0, 300]
        self.assertEqual(
            cocoGt.get_ann_ids(img_ids=[image_id], area_rng=areaRng, iscrowd=False),
            [
                key
                for key, val in cocoGt.anns.items()
                if (val["area"] > areaRng[0] and val["area"] < areaRng[1]) and (int(val["image_id"]) == image_id)
            ],
        )

    def test_getCatIds(self):
        cocoGt = COCO(self.gt_file)
        self.assertEqual(cocoGt.getCatIds(), list(cocoGt.cats.keys()))

        category_name = "acorn"
        self.assertEqual(
            cocoGt.get_cat_ids(cat_names=[category_name]),
            [key for key, val in cocoGt.cats.items() if val["name"] == category_name],
        )

        category_id = 191
        self.assertEqual(
            cocoGt.get_cat_ids(cat_ids=[category_id]),
            [key for key, val in cocoGt.cats.items() if val["id"] == category_id],
        )

    def test_getImgIds(self):
        cocoGt = COCO(self.gt_file)
        self.assertEqual(cocoGt.getImgIds(), list(cocoGt.imgs.keys()))

        category_id = 191

        self.assertEqual(
            cocoGt.get_img_ids(cat_ids=[category_id]),
            list(set([val["image_id"] for _, val in cocoGt.anns.items() if int(val["category_id"]) == category_id])),
        )

        self.assertEqual(
            cocoGt.get_img_ids(cat_ids=[category_id], img_ids=[121744]),
            [121744],
        )

    def test_load_all(self):
        cocoGt = COCO(self.gt_file)

        with self.assertRaises(TypeError):
            self.assertEqual(cocoGt.load_anns(), list(cocoGt.imgs.keys()))

        with self.assertRaises(TypeError):
            self.assertEqual(cocoGt.load_imgs(), list(cocoGt.imgs.keys()))

        with self.assertRaises(TypeError):
            self.assertEqual(cocoGt.load_cats(), list(cocoGt.cats.keys()))

        image_id = 521509
        self.assertEqual(cocoGt.load_imgs(ids=image_id), [cocoGt.imgs[521509]])

        self.assertEqual(cocoGt.load_imgs(ids=[image_id]), [cocoGt.imgs[521509]])

        category_id = 191
        self.assertEqual(cocoGt.load_cats(ids=category_id), [cocoGt.cats[category_id]])
        self.assertEqual(cocoGt.load_cats(ids=[category_id]), [cocoGt.cats[category_id]])

        ann_id = 1
        self.assertEqual(cocoGt.load_anns(ids=ann_id), [cocoGt.anns[ann_id]])


if __name__ == "__main__":
    unittest.main()
