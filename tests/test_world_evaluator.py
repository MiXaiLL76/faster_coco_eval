import copy
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

from faster_coco_eval import COCO

try:
    import torch
    import torch.distributed as dist

    from faster_coco_eval.utils.pytorch import FasterCocoEvaluator
except ImportError:
    raise unittest.SkipTest("Skipping all tests for World COCO Evaluator.")

TESTS_DIR = Path(__file__).parent


class TestWorldCoco(unittest.TestCase):
    """Test basic rankX COCO functionality."""

    maxDiff = None

    def setUp(self):
        """Prepare evaluation fixtures and an isolated Gloo rendezvous path."""
        self.gt_lvis_file = TESTS_DIR / "lvis_dataset" / "lvis_val_100.json"
        self.dt_lvis_file = TESTS_DIR / "lvis_dataset" / "lvis_results_100.json"
        self._rendezvous_dir = tempfile.TemporaryDirectory()
        self._rendezvous_path = Path(self._rendezvous_dir.name) / "gloo-rendezvous"
        # Only a process group initialized by this test may be destroyed here.
        self._owns_process_group = False

        # Regression pin, recorded 2026-07-22 against faster_coco_eval 1.7.2.
        # Cross-check with the official LVIS API using scripts/derive_lvis_golden.py.
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

    def tearDown(self):
        """Release distributed resources and the isolated rendezvous
        directory."""
        try:
            if self._owns_process_group and dist.is_initialized():
                dist.destroy_process_group()
        finally:
            self._rendezvous_dir.cleanup()

    def test_tear_down_destroys_initialized_process_group(self):
        """Destroy the default process group after a distributed test."""
        self._owns_process_group = True
        with mock.patch.object(dist, "destroy_process_group") as destroy_process_group:
            with mock.patch.object(dist, "is_initialized", return_value=True):
                self.tearDown()

        destroy_process_group.assert_called_once_with()

    def test_tear_down_preserves_unowned_process_group(self):
        """Avoid destroying a process group created outside this test case."""
        with mock.patch.object(dist, "destroy_process_group") as destroy_process_group:
            with mock.patch.object(dist, "is_initialized", return_value=True):
                self.tearDown()

        destroy_process_group.assert_not_called()

    def test_world_lvis(self):
        """Evaluate LVIS predictions through an isolated single-process
        group."""
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
        # File rendezvous avoids sharing a TCP port with parallel test workers.
        dist.init_process_group("gloo", rank=0, world_size=world_size, init_method=self._rendezvous_path.as_uri())
        self._owns_process_group = True

        for image_id, data in predictions.items():
            coco_eval_rank.update({image_id: data})

        coco_eval_rank.synchronize_between_processes()
        coco_eval_rank.accumulate()
        coco_eval_rank.summarize()

        actual_stats = coco_eval_rank.stats_as_dict["bbox"]
        self.assertEqual(actual_stats.keys(), self.stats_as_dict_result.keys())
        for key, expected_value in self.stats_as_dict_result.items():
            self.assertAlmostEqual(actual_stats[key], expected_value, places=10, msg=key)


if __name__ == "__main__":
    unittest.main()
