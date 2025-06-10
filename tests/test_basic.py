#!/usr/bin/python3
import os
import unittest

import numpy as np

import faster_coco_eval
import faster_coco_eval.core.mask as mask_util
from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.core.cocoeval import Params
from faster_coco_eval.core.faster_eval_api import COCOeval
from faster_coco_eval.extra import PreviewResults


class TestBaseCoco(unittest.TestCase):
    """Test basic COCO functionality."""

    maxDiff = None

    def setUp(self):
        self.root_folder = "dataset"

        self.gt_file = os.path.join(self.root_folder, "gt_dataset.json")
        self.dt_file = os.path.join(self.root_folder, "dt_dataset.json")
        self.gt_ignore_test_file = os.path.join(self.root_folder, "gt_ignore_test.json")
        self.dt_ignore_test_file = os.path.join(self.root_folder, "dt_ignore_test.json")

        if not os.path.exists(self.gt_file):
            self.root_folder = os.path.join(os.path.dirname(__file__), "dataset")

            self.gt_file = os.path.join(self.root_folder, "gt_dataset.json")
            self.dt_file = os.path.join(self.root_folder, "dt_dataset.json")
            self.gt_ignore_test_file = os.path.join(self.root_folder, "gt_ignore_test.json")
            self.dt_ignore_test_file = os.path.join(self.root_folder, "dt_ignore_test.json")

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

    def test_params(self):
        kp_params = Params(iouType="keypoints", kpt_sigmas=[1, 2, 3])
        kp_params.iouThrs = [0.5, 0.6, 0.7]
        kp_params.recThrs = [0.8, 0.9, 1.0]
        kp_params.areaRng = [[0, 100**2], [100**2, 200**2]]
        kp_params.imgIds = [1, 2, 3]
        self.assertEqual(kp_params.kpt_oks_sigmas.tolist(), [1, 2, 3])
        self.assertEqual(kp_params.iou_type, "keypoints")
        self.assertEqual(kp_params.iou_thrs, [0.5, 0.6, 0.7])
        self.assertEqual(kp_params.rec_thrs, [0.8, 0.9, 1.0])
        self.assertEqual(kp_params.max_dets, [20])
        self.assertEqual(kp_params.use_cats, 1)
        self.assertEqual(kp_params.area_rng, [[0, 100**2], [100**2, 200**2]])
        self.assertEqual(kp_params.img_ids, [1, 2, 3])
        self.assertEqual(kp_params.area_rng_lbl, ["all", "medium", "large"])

        kp_params.useSegm = 1
        self.assertEqual(kp_params.iou_type, "segm")

    def test_bad_coco_set(self):
        with self.assertRaises(TypeError):
            COCO(1)

    def test_bad_iou_type(self):
        with self.assertRaises(TypeError):
            ignore_prepared_anns = COCO.load_json(self.gt_ignore_test_file)
            cocoGt = COCO(ignore_prepared_anns)
            cocoDt = cocoGt.loadRes(self.dt_ignore_test_file)
            COCOeval_faster(cocoGt, cocoDt, "iouType")

    def test_ignore_coco_eval(self):
        stats_as_dict = {
            "AP_all": 0.7099009900990099,
            "AP_50": 1.0,
            "AP_75": 0.8415841584158416,
            "AP_small": -1.0,
            "AP_medium": 0.7099009900990099,
            "AP_large": -1.0,
            "AR_all": 0.0,
            "AR_second": 0.45384615384615384,
            "AR_third": 0.7153846153846154,
            "AR_small": -1.0,
            "AR_medium": 0.7153846153846154,
            "AR_large": -1.0,
            "AR_50": 1.0,
            "AR_75": 0.8461538461538461,
        }

        iouType = "bbox"

        ignore_prepared_anns = COCO.load_json(self.gt_ignore_test_file)
        cocoGt = COCO(ignore_prepared_anns, use_deepcopy=True)

        self.assertNotEqual(id(ignore_prepared_anns), id(cocoGt.dataset))

        cocoDt = cocoGt.loadRes(self.dt_ignore_test_file, min_score=0.1)

        cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType, separate_eval=True)

        cocoEval.run()

        for imgId, catId in cocoEval.ious:
            np.testing.assert_array_equal(
                cocoEval.ious[(imgId, catId)], np.load(os.path.join(self.root_folder, f"{imgId}_{catId}.npy"))
            )

        for key in stats_as_dict:
            self.assertAlmostEqual(cocoEval.stats_as_dict[key], stats_as_dict[key], places=10, msg=key)

    def test_coco_eval(self):
        stats_as_dict = {
            "AP_all": 0.6947194719471946,
            "AP_50": 0.6947194719471946,
            "AP_75": 0.6947194719471946,
            "AP_small": -1.0,
            "AP_medium": 0.7367986798679867,
            "AP_large": 0.0,
            "AR_all": 0.6666666666666666,
            "AR_second": 0.75,
            "AR_third": 0.75,
            "AR_small": -1.0,
            "AR_medium": 0.7916666666666666,
            "AR_large": 0.0,
            "AR_50": 0.75,
            "AR_75": 0.75,
            "mIoU": 1.0,
            "mAUC_50": 0.8148148148148149,
        }

        cocoGt = COCO(self.gt_file)

        cocoDt = cocoGt.loadRes(self.prepared_anns_droped_bbox)

        # iouType="segm" as default!
        cocoEval = COCOeval(cocoGt, cocoDt, extra_calc=True)
        self.assertEqual(cocoEval.params.useSegm, 1)

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        self.assertEqual(cocoEval.matched, True)
        self.assertAlmostEqual(cocoEval.stats_as_dict, stats_as_dict, places=10)

    def test_confusion_matrix(self):
        prepared_result = [
            [2.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        ]

        iouType = "segm"
        useCats = False

        cocoGt = COCO(self.gt_file)
        cocoDt = cocoGt.loadRes(self.dt_file)

        results = PreviewResults(
            cocoGt=cocoGt,
            cocoDt=cocoDt,
            iouType=iouType,
            iou_tresh=0.5,
            useCats=useCats,
        )

        self.assertEqual(results.cocoEval.matched, True)

        result_cm = results.compute_confusion_matrix().tolist()

        self.assertEqual(result_cm, prepared_result)

    def test_rerp(self):
        cocoGt = COCO(self.gt_file)
        cocoDt = cocoGt.loadRes(self.prepared_anns_droped_bbox)
        cocoEval = COCOeval_faster(cocoGt, cocoDt, extra_calc=True)

        self.assertEqual(
            repr(cocoGt),
            f"COCO(annotation_file) # __author__='{faster_coco_eval.__author__}';"
            f" __version__='{faster_coco_eval.__version__}';",
        )
        self.assertEqual(
            repr(cocoEval),
            f"COCOeval_faster() # __author__='{faster_coco_eval.__author__}';"
            f" __version__='{faster_coco_eval.__version__}';",
        )

    def test_auc(self):
        x = np.linspace(0, 0.55, 100)
        y = np.linspace(0, 2, 100) + 0.1

        cpp_auc = COCOeval_faster.calc_auc(x, y)
        py_auc = COCOeval_faster.calc_auc(x, y, method="py")
        # sklearn not in test space!
        # from sklearn import metrics
        # orig_auc = metrics.auc(x, y)
        orig_auc = 1.1550000000000005

        self.assertAlmostEqual(cpp_auc, orig_auc, places=8)
        self.assertAlmostEqual(py_auc, orig_auc, places=8)

    def test_to_dict(self):
        orig_data = COCO.load_json(self.gt_file)
        cocoGt = COCO(self.gt_file)

        parsed_data = cocoGt.to_dict()

        self.assertListEqual(parsed_data["annotations"], orig_data["annotations"])
        self.assertListEqual(parsed_data["categories"], orig_data["categories"])
        self.assertListEqual(parsed_data["images"], orig_data["images"])

        parsed_data = dict(cocoGt)

        self.assertListEqual(parsed_data["annotations"], orig_data["annotations"])
        self.assertListEqual(parsed_data["categories"], orig_data["categories"])
        self.assertListEqual(parsed_data["images"], orig_data["images"])


if __name__ == "__main__":
    unittest.main()
