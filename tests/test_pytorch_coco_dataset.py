import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torchvision
except ImportError:
    raise unittest.SkipTest("Skipping PyTorch dataset tests because torchvision is unavailable.")

import faster_coco_eval
from faster_coco_eval.utils.pytorch import coco_dataset


def test_import_does_not_patch_pycocotools():
    """Importing the dataset module must not mutate process-wide modules."""
    with patch.object(faster_coco_eval, "init_as_pycocotools") as init_as_pycocotools:
        importlib.reload(coco_dataset)

    init_as_pycocotools.assert_not_called()


def test_constructor_patches_before_base_parse_and_reuses_base_coco():
    """Initialize the compatibility module before torchvision parses
    annotations."""

    def fake_base_init(dataset, *args, **kwargs):
        """Provide the COCO object that torchvision normally creates."""
        dataset.coco = SimpleNamespace(imgs={2: {}})

    with (
        patch.object(faster_coco_eval, "init_as_pycocotools") as init_as_pycocotools,
        patch.object(
            torchvision.datasets.CocoDetection, "__init__", autospec=True, side_effect=fake_base_init
        ) as base_init,
        patch.object(coco_dataset, "COCO", create=True) as explicit_coco,
    ):
        dataset = coco_dataset.FasterCocoDetection(
            "root",
            "annotations.json",
            transform="transform",
            target_transform="target",
            transforms="transforms",
        )

    init_as_pycocotools.assert_called_once_with()
    base_init.assert_called_once_with(
        dataset,
        "root",
        "annotations.json",
        "transform",
        "target",
        "transforms",
    )
    explicit_coco.assert_not_called()
    assert dataset.ids == [2]
