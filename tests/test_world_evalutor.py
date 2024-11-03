#!/usr/bin/python3
import concurrent.futures
import copy
import itertools
import os
import unittest
from collections import defaultdict

import numpy as np

import faster_coco_eval.core.mask as mask_util
from faster_coco_eval import COCO, COCOeval_faster


def _evaluate(coco_eval_proc: COCOeval_faster, anns: list):
    coco_eval_proc.params.imgIds = [ann["image_id"] for ann in anns]
    coco_eval_proc.cocoDt = coco_eval_proc.cocoGt.loadRes(anns)
    coco_eval_proc.evaluate()
    return coco_eval_proc._evalImgs_cpp, coco_eval_proc.params.imgIds


class TestWorldCoco(unittest.TestCase):
    """Test basic rankX COCO functionality."""

    def setUp(self):
        self.gt_file = os.path.join("dataset", "gt_dataset.json")
        self.dt_file = os.path.join("dataset", "dt_dataset.json")
        self.gt_lvis_file = os.path.join("lvis_dataset", "lvis_val_100.json")
        self.dt_lvis_file = os.path.join("lvis_dataset", "lvis_results_100.json")

        if not os.path.exists(self.gt_file):
            self.gt_file = os.path.join(os.path.dirname(__file__), self.gt_file)
            self.dt_file = os.path.join(os.path.dirname(__file__), self.dt_file)
            self.gt_lvis_file = os.path.join(os.path.dirname(__file__), self.gt_lvis_file)
            self.dt_lvis_file = os.path.join(os.path.dirname(__file__), self.dt_lvis_file)

        prepared_anns = COCO.load_json(self.dt_file)

        cocoGt = COCO(self.gt_file)
        self.prepared_anns_droped_bbox = [
            {
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "iscrowd": ann["iscrowd"],
                "id": ann["id"],
                "score": ann["score"],
                "segmentation": mask_util.merge(
                    mask_util.frPyObjects(
                        ann["segmentation"],
                        cocoGt.imgs[ann["image_id"]]["height"],
                        cocoGt.imgs[ann["image_id"]]["width"],
                    )
                ),
            }
            for ann in prepared_anns
        ]

        self.prepared_anns_droped_bbox_by_image_id = defaultdict(list)
        for ann in self.prepared_anns_droped_bbox:
            self.prepared_anns_droped_bbox_by_image_id[ann["image_id"]].append(copy.deepcopy(ann))

    def test_world(self):
        # MULTI
        coco_eval_rank = COCOeval_faster(COCO(self.gt_file), iouType="segm", separate_eval=True)

        eval_imgs = []
        eval_img_ids = []

        with concurrent.futures.ProcessPoolExecutor(len(self.prepared_anns_droped_bbox_by_image_id)) as executor:
            for evalImgs, imgIds in executor.map(
                _evaluate, itertools.repeat(coco_eval_rank), self.prepared_anns_droped_bbox_by_image_id.values()
            ):
                eval_imgs.append(evalImgs)
                eval_img_ids += imgIds

        coco_eval_rank.params.imgIds = eval_img_ids
        coco_eval_rank._paramsEval = copy.deepcopy(coco_eval_rank.params)

        coco_eval_rank._evalImgs_cpp = np.array(eval_imgs).T.ravel().tolist()

        coco_eval_rank.accumulate()
        coco_eval_rank.summarize()

        # SOLO
        coco_eval = COCOeval_faster(COCO(self.gt_file), iouType="segm")
        coco_eval.cocoDt = coco_eval.cocoGt.loadRes(self.prepared_anns_droped_bbox)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        self.assertAlmostEqual(coco_eval.stats_as_dict, coco_eval_rank.stats_as_dict, 12)

    def test_world_lvis(self):
        # MULTI
        coco_eval_rank = COCOeval_faster(COCO(self.gt_lvis_file), iouType="bbox", lvis_style=True, separate_eval=True)

        prepared_anns = defaultdict(list)
        for ann in COCO.load_json(self.dt_lvis_file):
            prepared_anns[ann["image_id"]].append(copy.deepcopy(ann))

        all_images_ids = list(prepared_anns.keys())
        rank2_prepared_anns = defaultdict(list)

        for key in all_images_ids[::2]:
            rank2_prepared_anns["rank1"] += prepared_anns[key]

        for key in all_images_ids[1::2]:
            rank2_prepared_anns["rank2"] += prepared_anns[key]

        eval_imgs = []
        eval_img_ids = []

        with concurrent.futures.ProcessPoolExecutor(2) as executor:
            for evalImgs, imgIds in executor.map(
                _evaluate, itertools.repeat(coco_eval_rank), rank2_prepared_anns.values()
            ):
                eval_imgs.append(evalImgs)
                eval_img_ids += imgIds

        coco_eval_rank.params.imgIds = eval_img_ids
        coco_eval_rank._paramsEval = copy.deepcopy(coco_eval_rank.params)
        coco_eval_rank.freq_groups = coco_eval_rank._prepare_freq_group()
        coco_eval_rank._evalImgs_cpp = np.array(eval_imgs).T.ravel().tolist()

        coco_eval_rank.accumulate()
        coco_eval_rank.summarize()

        # SOLO
        coco_eval = COCOeval_faster(COCO(self.gt_lvis_file), iouType="bbox", lvis_style=True)
        coco_eval.cocoDt = coco_eval.cocoGt.loadRes(self.dt_lvis_file)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        self.assertAlmostEqual(coco_eval.stats_as_dict, coco_eval_rank.stats_as_dict, 12)


if __name__ == "__main__":
    unittest.main()
