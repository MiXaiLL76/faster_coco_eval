import unittest
from copy import deepcopy
from unittest import TestCase

from parameterized import parameterized

try:
    import torch
    from lightning_utilities import apply_to_collection
    from torch import BoolTensor, IntTensor, Tensor
    from torchmetrics import MetricCollection
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:
    raise unittest.SkipTest("Skipping all tests for torchmetrics.")

# fmt: off
_inputs = {
    "preds": [
        [
            {
                "boxes": Tensor([[258.15, 41.29, 606.41, 285.07]]),
                "scores": Tensor([0.236]),
                "labels": IntTensor([4]),
            },  # coco image id 42
            {
                "boxes": Tensor([
                    [61.00, 22.75, 565.00, 632.42],
                    [12.66, 3.32, 281.26, 275.23]]),
                "scores": Tensor([0.318, 0.726]),
                "labels": IntTensor([3, 2]),
            },  # coco image id 73
        ],
        [
            {
                "boxes": Tensor([
                    [87.87, 276.25, 384.29, 379.43],
                    [0.00, 3.66, 142.15, 316.06],
                    [296.55, 93.96, 314.97, 152.79],
                    [328.94, 97.05, 342.49, 122.98],
                    [356.62, 95.47, 372.33, 147.55],
                    [464.08, 105.09, 495.74, 146.99],
                    [276.11, 103.84, 291.44, 150.72],
                ]),
                "scores": Tensor([0.546, 0.3, 0.407,
                                 0.611, 0.335, 0.805, 0.953]),
                "labels": IntTensor([4, 1, 0, 0, 0, 0, 0]),
            },  # coco image id 74
            {
                "boxes": Tensor([
                    [72.92, 45.96, 91.23, 80.57],
                    [45.17, 45.34, 66.28, 79.83],
                    [82.28, 47.04, 99.66, 78.50],
                    [59.96, 46.17, 80.35, 80.48],
                    [75.29, 23.01, 91.85, 50.85],
                    [71.14, 1.10, 96.96, 28.33],
                    [61.34, 55.23, 77.14, 79.57],
                    [41.17, 45.78, 60.99, 78.48],
                    [56.18, 44.80, 64.42, 56.25],
                ]),
                "scores": Tensor([
                    0.532,
                    0.204,
                    0.782,
                    0.202,
                    0.883,
                    0.271,
                    0.561,
                    0.204 + 1e-8,  # There are some problems with sorting at the moment. When sorting score with the same values, they give different indexes. # noqa: E501
                    0.349
                ]),
                "labels": IntTensor([
                    49,
                    49,
                    49,
                    49,
                    49,
                    49,
                    49,
                    49,
                    49
                ]),
            },  # coco image id 987 category_id 49
        ],
    ],
    "target": [
        [
            {
                "boxes": Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]),
                "labels": IntTensor([4]),
            },  # coco image id 42
            {
                "boxes": Tensor([
                    [13.00, 22.75, 548.98, 632.42],
                    [1.66, 3.32, 270.26, 275.23],
                ]),
                "labels": IntTensor([2, 2]),
            },  # coco image id 73
        ],
        [
            {
                "boxes": Tensor([
                    [61.87, 276.25, 358.29, 379.43],
                    [2.75, 3.66, 162.15, 316.06],
                    [295.55, 93.96, 313.97, 152.79],
                    [326.94, 97.05, 340.49, 122.98],
                    [356.62, 95.47, 372.33, 147.55],
                    [462.08, 105.09, 493.74, 146.99],
                    [277.11, 103.84, 292.44, 150.72],
                ]),
                "labels": IntTensor([4, 1, 0, 0, 0, 0, 0]),
            },  # coco image id 74
            {
                "boxes": Tensor([
                    [72.92, 45.96, 91.23, 80.57],
                    [50.17, 45.34, 71.28, 79.83],
                    [81.28, 47.04, 98.66, 78.50],
                    [63.96, 46.17, 84.35, 80.48],
                    [75.29, 23.01, 91.85, 50.85],
                    [56.39, 21.65, 75.66, 45.54],
                    [73.14, 1.10, 98.96, 28.33],
                    [62.34, 55.23, 78.14, 79.57],
                    [44.17, 45.78, 63.99, 78.48],
                    [58.18, 44.80, 66.42, 56.25],
                ]),
                "labels": IntTensor([
                    49,
                    49,
                    49,
                    49,
                    49,
                    49,
                    49,
                    49,
                    49,
                    49
                ]),
            },  # coco image id 987 category_id 49
        ],
    ],
}
# fmt: on


class TestTorchmetricsLib(TestCase):
    maxDiff = None

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
        metric = MeanAveragePrecision(iou_type="bbox", backend="faster_coco_eval")

        # Update metric with predictions and respective ground truth
        metric.update(self.preds, self.target)

        # Compute the results
        result = metric.compute()

        self.assertDictEqual(result, self.valid_result)

    def test_segm_iou_empty_gt_mask(self):
        """Test empty ground truths."""
        backend = "faster_coco_eval"
        metric = MeanAveragePrecision(iou_type="segm", backend=backend)
        metric.update(
            [
                {
                    "masks": torch.randint(0, 1, (1, 10, 10)).bool(),
                    "scores": Tensor([0.5]),
                    "labels": IntTensor([4]),
                }
            ],
            [{"masks": Tensor([]), "labels": IntTensor([])}],
        )
        metric.compute()

    @parameterized.expand([False, True])
    def test_average_argument(self, class_metrics):
        """Test that average argument works.

        Calculating macro on inputs that only have one label should be
        the same as micro. Calculating class metrics should be the same
        regardless of average argument.
        """
        backend = "pycocotools"
        backend = "faster_coco_eval"

        if class_metrics:
            _preds = _inputs["preds"]
            _target = _inputs["target"]
        else:
            _preds = apply_to_collection(
                deepcopy(_inputs["preds"]),
                IntTensor,
                lambda x: torch.ones_like(x),
            )
            _target = apply_to_collection(
                deepcopy(_inputs["target"]),
                IntTensor,
                lambda x: torch.ones_like(x),
            )

        metric_macro = MeanAveragePrecision(average="macro", class_metrics=class_metrics, backend=backend)
        metric_macro.update(_preds[0], _target[0])
        metric_macro.update(_preds[1], _target[1])
        result_macro = metric_macro.compute()

        metric_micro = MeanAveragePrecision(average="micro", class_metrics=class_metrics, backend=backend)
        metric_micro.update(_inputs["preds"][0], _inputs["target"][0])
        metric_micro.update(_inputs["preds"][1], _inputs["target"][1])
        result_micro = metric_micro.compute()

        if class_metrics:
            assert torch.allclose(result_macro["map_per_class"], result_micro["map_per_class"])
            assert torch.allclose(
                result_macro["mar_100_per_class"],
                result_micro["mar_100_per_class"],
            )
        else:
            for key in result_macro:
                if key == "classes":
                    continue
                assert torch.allclose(result_macro[key], result_micro[key])

    def test_metric_collection(self):
        metrics = MetricCollection([MeanAveragePrecision(average="macro", backend="faster_coco_eval")])
        result = metrics(self.preds, self.target)
        self.assertDictEqual(result, self.valid_result)


if __name__ == "__main__":
    unittest.main()
