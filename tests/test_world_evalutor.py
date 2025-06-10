import copy
import os
import unittest
from collections import defaultdict

from faster_coco_eval import COCO

try:
    import torch
    import torch.distributed as dist
except ImportError:
    raise unittest.SkipTest("Skipping all tests for World COCO Evaluator.")

from faster_coco_eval.utils.pytorch import FasterCocoEvaluator


class TestWorldCoco(unittest.TestCase):
    """Test basic rankX COCO functionality."""

    maxDiff = None

    def setUp(self):
        self.gt_lvis_file = os.path.join("lvis_dataset", "lvis_val_100.json")
        self.dt_lvis_file = os.path.join("lvis_dataset", "lvis_results_100.json")

        if not os.path.exists(self.gt_lvis_file):
            self.gt_lvis_file = os.path.join(os.path.dirname(__file__), self.gt_lvis_file)
            self.dt_lvis_file = os.path.join(os.path.dirname(__file__), self.dt_lvis_file)

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

    def test_world_lvis(self):
        coco_gt = COCO(self.gt_lvis_file)
        coco_eval_rank = FasterCocoEvaluator(coco_gt, iou_types=["bbox"], lvis_style=True)
        coco_eval_rank.coco_eval["bbox"].params.maxDets = [300]

        prepared_anns = defaultdict(list)
        for ann in COCO.load_json(self.dt_lvis_file):
            prepared_anns[ann["image_id"]].append(copy.deepcopy(ann))

        predictions = {}
        for image_id, anns in prepared_anns.items():
            boxes = torch.Tensor([ann["bbox"] for ann in anns])
            boxes[:, 2:] += boxes[:, :2]

            predictions[image_id] = {
                "boxes": boxes,
                "scores": torch.Tensor([ann["score"] for ann in anns]),
                "labels": torch.Tensor([ann["category_id"] for ann in anns]),
            }

        world_size = 1
        dist.init_process_group("gloo", rank=0, world_size=world_size, init_method="tcp://127.0.0.1:1234")

        for image_id, data in predictions.items():
            coco_eval_rank.update({image_id: data})

        coco_eval_rank.synchronize_between_processes()
        coco_eval_rank.accumulate()
        coco_eval_rank.summarize()

        self.assertEqual(coco_eval_rank.stats_as_dict["bbox"], self.stats_as_dict_result)


if __name__ == "__main__":
    unittest.main()
