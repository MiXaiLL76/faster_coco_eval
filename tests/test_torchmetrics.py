import unittest
from unittest import TestCase

try:
    import torch
    from torch import BoolTensor, IntTensor, Tensor
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:
    raise unittest.SkipTest("Skipping all tests for torchmetrics.")


class TestTorchmetricsLib(TestCase):
    def setUp(self):
        # Preds should be a list of elements, where each element is a dict
        # containing 3 keys: boxes, scores, labels
        mask_pred = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        self.preds = [
            {
                # The boxes keyword should contain an [N,4] tensor,
                # where N is the number of detected \
                #   boxes with boxes of the format
                # [xmin, ymin, xmax, ymax] in absolute \
                #   image coordinates
                "boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]),
                # The scores keyword should contain an [N,] \
                #   tensor where
                # each element is confidence score between 0 and 1
                "scores": Tensor([0.536]),
                # The labels keyword should contain an [N,] tensor
                # with integers of the predicted classes
                "labels": IntTensor([0]),
                # The masks keyword should contain an [N,H,W] tensor,
                # where H and W are the image height and width, \
                #   respectively,
                # with boolean masks. This is only required \
                #   when iou_type is `segm`.
                "masks": BoolTensor([mask_pred]),
            }
        ]

        # Target should be a list of elements, where each element is a dict
        # containing 2 keys: boxes and labels \
        #   (and masks, if iou_type is `segm`).
        # Each keyword should be formatted similar to the preds argument.
        # The number of elements in preds and target need to match
        mask_tgt = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        self.target = [
            {
                "boxes": Tensor([[214.0, 41.0, 562.0, 285.0]]),
                "labels": IntTensor([0]),
                "masks": BoolTensor([mask_tgt]),
            }
        ]

        self.valid_result = {
            "map": torch.tensor(0.6000),
            "map_50": torch.tensor(1.0),
            "map_75": torch.tensor(1.0),
            "map_small": torch.tensor(-1.0),
            "map_medium": torch.tensor(-1.0),
            "map_large": torch.tensor(0.6000),
            "mar_1": torch.tensor(0.6000),
            "mar_10": torch.tensor(0.6000),
            "mar_100": torch.tensor(0.6000),
            "mar_small": torch.tensor(-1.0),
            "mar_medium": torch.tensor(-1.0),
            "mar_large": torch.tensor(0.6000),
            "map_per_class": torch.tensor(-1.0),
            "mar_100_per_class": torch.tensor(-1.0),
            "classes": torch.tensor(0, dtype=torch.int32),
        }

    def test_evaluate(self):
        # Initialize metric
        metric = MeanAveragePrecision(
            iou_type="bbox", backend="faster_coco_eval"
        )

        # Update metric with predictions and respective ground truth
        metric.update(self.preds, self.target)

        self.assertEqual(metric.backend, "faster_coco_eval")

        # Compute the results
        result = metric.compute()

        self.assertDictEqual(result, self.valid_result)


if __name__ == "__main__":
    unittest.main()
