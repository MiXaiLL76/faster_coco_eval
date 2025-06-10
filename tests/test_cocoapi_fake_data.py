#!/usr/bin/python3
import pathlib
import unittest

from parameterized import parameterized

from faster_coco_eval import COCO, COCOeval_faster


def get_data_files(root_folder: pathlib.PosixPath = pathlib.Path("dataset/cocoapi")):
    _DETECTION_VAL = root_folder / "instances_val2014_100.json"
    _DETECTION_BBOX = root_folder / "instances_val2014_fakebbox100_results.json"
    _DETECTION_SEGM = root_folder / "instances_val2014_fakesegm100_results.json"

    return {"val": _DETECTION_VAL, "bbox": _DETECTION_BBOX, "segm": _DETECTION_SEGM}


# https://github.com/cocodataset/cocoapi/tree/master/results
class TestCOCOAPI(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.root_folder = pathlib.Path("dataset/cocoapi")

        if not self.root_folder.exists():
            self.root_folder = pathlib.Path(__file__).parent / self.root_folder

        self.data_files = get_data_files(self.root_folder)

        self.stats = {
            "keys": [
                "AP_all",
                "AP_50",
                "AP_75",
                "AP_small",
                "AP_medium",
                "AP_large",
                "AR_all",
                "AR_second",
                "AR_third",
                "AR_small",
                "AR_medium",
                "AR_large",
            ],
            "bbox": [
                0.5045806987249628,
                0.6969727247299577,
                0.5729816669904824,
                0.5856257209410443,
                0.5193996948036719,
                0.5013978986347466,
                0.38681277964578054,
                0.5936795762842003,
                0.595352982877607,
                0.6398109626113442,
                0.5664205978994309,
                0.5642905982905982,
            ],
            "segm": [
                0.3195452758576433,
                0.5622883972521636,
                0.29892653412086784,
                0.3873740315997838,
                0.31018272403369485,
                0.3269339071005138,
                0.2682297225711534,
                0.41544868114906375,
                0.4168394992198818,
                0.4694498622754236,
                0.37675922666197265,
                0.3814715099715099,
            ],
        }

    @parameterized.expand(["bbox", "segm"])
    def test_eval(self, iouType: str):
        cocoGt = COCO(self.data_files["val"])

        cocoDt = cocoGt.loadRes(self.data_files[iouType])

        cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType=iouType)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        for idx, key in enumerate(self.stats["keys"]):
            self.assertAlmostEqual(cocoEval.stats[idx], self.stats[iouType][idx], places=10, msg=key)


if __name__ == "__main__":
    unittest.main()
