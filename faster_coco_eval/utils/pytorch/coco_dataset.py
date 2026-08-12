"""MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

Mostly copy-paste from
https://github.com/pytorch/vision/blob/edfd5a7701310589927d2f83bed11cfeb06965a1/torchvision/datasets/coco.py
The difference is that pycocotools is replaced by a faster library faster-coco-eval
"""

from collections.abc import Callable
from pathlib import Path

import torchvision

import faster_coco_eval


class FasterCocoDetection(torchvision.datasets.CocoDetection):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (str or Path): Root directory where images are downloaded to.
        annFile (str): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g., ``transforms.ToTensor``.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str | Path,
        annFile: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
    ) -> None:
        """Initializes the FasterCocoDetection dataset.

        Args:
            root (str or Path): Root directory where images are downloaded to.
            annFile (str): Path to json annotation file.
            transform (Callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version. Default is None.
            target_transform (Callable, optional): A function/transform that takes in the
                target and transforms it. Default is None.
            transforms (Callable, optional): A function/transform that takes input sample and its target as entry
                and returns a transformed version. Default is None.

        Returns:
            None
        """
        faster_coco_eval.init_as_pycocotools()
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.ids = list(sorted(self.coco.imgs.keys()))
