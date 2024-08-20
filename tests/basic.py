#!/usr/bin/python3

import io
import os
import copy
import unittest
from contextlib import redirect_stdout

import numpy as np

import faster_coco_eval.core.mask as mask_util
from faster_coco_eval.core.boundary_utils import add_boundary_multi_core
from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.extra import PreviewResults


def _encode(x):
    """Encode a binary mask into a run-length encoded string."""
    return mask_util.encode(np.asfortranarray(x, np.uint8))


class TestBaseCoco(unittest.TestCase):
    """Test basic COCO functionality."""

    prepared_coco_in_dict = None
    prepared_anns = None

    def setUp(self):
        gt_file = "dataset/gt_dataset.json"
        dt_file = "dataset/dt_dataset.json"

        if not os.path.exists(gt_file):
            gt_file = os.path.join("tests", gt_file)
            dt_file = os.path.join("tests", dt_file)

        self.prepared_coco_in_dict = COCO.load_json(gt_file)
        self.prepared_anns = COCO.load_json(dt_file)

    def test_coco_eval(self):
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

        cocoGt = COCO(self.prepared_coco_in_dict)
        cocoDt = cocoGt.loadRes(self.prepared_anns)

        cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType, extra_calc=True)
        cocoEval.params.maxDets = [len(cocoGt.anns)]

        if not useCats:
            cocoEval.params.useCats = 0  # Выключение labels

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        self.assertEqual(cocoEval.matched, True)
        self.assertEqual(cocoEval.stats_as_dict, stats_as_dict)

    def test_boundary_eval(self):
        stats_as_dict = {
            # the following values (except for mIoU and mAUC_50) have been
            # obtained by running the original boundary_iou_api on
            # gt_dataset and dt_dataset
            "AP_all": 0.6947194719471946,
            "AP_50": 0.6947194719471946,
            "AP_75": 0.6947194719471946,
            "AP_small": -1.0,
            "AP_medium": 0.7920792079207921,
            "AP_large": 0.0,
            "AR_1": 0.6666666666666666,
            "AR_10": 0.75,
            "AR_100": 0.75,
            "AR_small": -1.0,
            "AR_medium": 0.7916666666666666,
            "AR_large": 0.0,
            "AR_50": 0.75,
            "AR_75": 0.75,
            "mIoU": 1.0,
            "mAUC_50": 0.8518518518518519
        }

        iouType = "boundary"

        cocoGt = COCO(self.prepared_coco_in_dict)
        cocoDt = cocoGt.loadRes(self.prepared_anns)
        with redirect_stdout(io.StringIO()):
            add_boundary_multi_core(cocoGt)
            add_boundary_multi_core(cocoDt)

        cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType, extra_calc=True)

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        self.assertEqual(cocoEval.stats_as_dict, stats_as_dict)

    def test_gts_as_dts(self):
        # a simple sanity check
        stats_as_dict = {
            "AP_all": 1.0,
            "AP_50": 1.0,
            "AP_75": 1.0,
            "AP_small": -1.0,
            "AP_medium": 1.0,
            "AP_large": 1.0,
            "AR_1": 1.0,
            "AR_10": 1.0,
            "AR_100": 1.0,
            "AR_small": -1.0,
            "AR_medium": 1.0,
            "AR_large": 1.0,
            "AR_50": 1.0,
            "AR_75": 1.0,
            "mIoU": 1.0,
            "mAUC_50": 1.0
        }
        # load gt annotations as predictions
        gts_as_dts = copy.deepcopy(
            self.prepared_coco_in_dict
            )['annotations']
        for p in gts_as_dts:
            p['score'] = 1  # fake score

        for iouType in ['bbox', 'segm', 'boundary']:
            cocoGt = COCO(self.prepared_coco_in_dict)
            cocoDt = cocoGt.loadRes(gts_as_dts)
            if iouType == 'boundary':
                with redirect_stdout(io.StringIO()):
                    add_boundary_multi_core(cocoGt)
                    add_boundary_multi_core(cocoDt)
            cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType,
                                       extra_calc=True)
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            self.assertEqual(cocoEval.stats_as_dict, stats_as_dict)

    def test_confusion_matrix(self):
        prepared_result = [
            [2.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        ]

        iouType = "segm"
        useCats = False

        cocoGt = COCO(self.prepared_coco_in_dict)
        cocoDt = cocoGt.loadRes(self.prepared_anns)

        results = PreviewResults(
            cocoGt=cocoGt,
            cocoDt=cocoDt,
            iouType=iouType,
            iou_tresh=0.5,
            useCats=useCats,
        )
        result_cm = results.compute_confusion_matrix().tolist()

        self.assertEqual(result_cm, prepared_result)

    def testToBboxFullImage(self):
        mask = np.array([[0, 1], [1, 1]])
        bbox = mask_util.toBbox(_encode(mask))
        self.assertTrue(
            (bbox == np.array([0, 0, 2, 2], dtype="float32")).all(), bbox
        )

    # bugfix by piotr in ff4a47150bf66
    def testToBboxNonFullImage(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:4, 3:6] = 1
        bbox = mask_util.toBbox(_encode(mask))
        self.assertTrue(
            (bbox == np.array([3, 2, 3, 2], dtype="float32")).all(), bbox
        )

    def testInvalidRLECounts(self):
        rle = {
            "size": [1024, 1024],
            "counts": "jd`0=`o06J5L4M3L3N2N2N2N2N1O2N2N101N1O2O0O1O2N100O1O2N100O1O1O1O1O101N1O1O1O1O1O1O101N1O100O101O0O100000000000000001O00001O1O0O2O1N3N1N3N3L5Kh0XO6J4K5L5Id[o5N]dPJ7K4K4M3N2M3N2N1O2N100O2O0O1000O01000O101N1O1O2N2N2M3M3M4J7Inml5H[RSJ6L2N2N2N2O000000000000O2O1N2N2Mkm81SRG6L3L3N2O1N2N2O0O2O00001O0000000000O2O001N2O0O2N2N3M3L5JRjf6MPVYI8J4L3N3M2N1O2O1N101N1000000O10000001O000O101N101N1O2N2N2N3L4L7FWZ_50ne`J0000001O000000001O0000001O1O0N3M3N1O2N2N2O1N2O001N2`RO^O`k0c0[TOEak0;\\\\TOJbk07\\\\TOLck03[TO0dk01ZTO2dk0OYTO4gk0KXTO7gk0IXTO8ik0HUTO:kk0ETTO=lk0CRTO>Pl0@oSOb0Rl0\\\\OmSOe0Tl0[OjSOg0Ul0YOiSOi0Wl0XOgSOi0Yl0WOeSOk0[l0VOaSOn0kh0cNmYO",  # noqa: E501
        }
        with self.assertRaises(ValueError):
            mask_util.decode(rle)

    def testZeroLeadingRLE(self):
        # A foreground segment of length 0 was not previously handled correctly.
        # This input rle has 3 leading zeros.
        rle = {
            "size": [1350, 1080],
            "counts": "000lg0Zb01O00001O00001O001O00001O00001O001O00001O01O2N3M3M3M2N3M3N2M3M2N1O1O1O1O2N1O1O1O2N1O1O101N1O1O1O2N1O1O1O2N3M2N1O2N1O2O0O2N1O1O2N1O2N1O2N1O2N1O2N1O2O0O2N1O3M2N1O2N2N2N2N2N1O2N2N2N2N1O2N2N2N2N2N1N3N2N00O1O1O1O100000000000000O100000000000000001O0000001O00001O0O5L7I5K4L4L3M2N2N2N1O2m]OoXOm`0Sg0j^OVYOTa0lf0c^O]YO[a0ef0\\^OdYOba0bg0N2N2N2N2N2N2N2N2N2N2N2N2N2N2N2N2N3M2M4M2N3M2N3M2N3M2N3M2N3M2N3M2N3M2N3M2M4M2N2N3M2M4M2N2N3M2M3N3M2N3M2M3N3M2N2N3L3N2N3M2N3L3N2N3M5J4M3M4L3M3L5L3M3M4L3L4\\EXTOd6jo0K6J5K6I4M1O1O1O1N2O1O1O001N2O00001O0O101O000O2O00001N101N101N2N101N101N101N2O0O2O0O2O0O2O1N101N2N2O1N2O1N2O1N101N2O1N2O1N2O0O2O1N2N2O1N2O0O2O1N2O1N2N2N1N4M2N2M4M2N3L3N2N3L3N3L3N2N3L3N2N3L3M4L3M3M4L3M5K5K5K6J5K5K6J7I7I7Ibijn0",  # noqa: E501
        }
        orig_bbox = mask_util.toBbox(rle)
        mask = mask_util.decode(rle)
        rle_new = mask_util.encode(mask)
        new_bbox = mask_util.toBbox(rle_new)
        self.assertTrue(np.equal(orig_bbox, new_bbox).all())


if __name__ == "__main__":
    unittest.main()
