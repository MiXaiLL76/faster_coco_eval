import copy
import logging
from collections import defaultdict
from typing import Dict, Set

import numpy as np

from ..core import COCO, COCOeval_faster

logger = logging.getLogger(__name__)


class ExtraEval:
    """Extra evaluation for coco dataset."""

    def __init__(
        self,
        cocoGt: COCO = None,
        cocoDt: COCO = None,
        iouType: str = "bbox",
        min_score: float = 0,
        iou_tresh: float = 0.0,
        recall_count: int = 100,
        useCats: bool = False,
        kpt_oks_sigmas: list = None,
    ):
        """Initializes the ExtraEval object.

        Args:
            cocoGt (COCO, optional): Ground truth COCO object. Defaults to None.
            cocoDt (COCO, optional): Detection results COCO object. Defaults to None.
            iouType (str, optional): Type of IoU evaluation ('bbox', 'segm', 'keypoints'). Defaults to "bbox".
            min_score (float, optional): Minimum score threshold for detections. Defaults to 0.
            iou_tresh (float, optional): IoU threshold for evaluation. Defaults to 0.0.
            recall_count (int, optional): Number of recall thresholds. Defaults to 100.
            useCats (bool, optional): Whether to use categories in evaluation. Defaults to False.
            kpt_oks_sigmas (list, optional): List of OKS sigmas for keypoints evaluation. Defaults to None.

        Raises:
            AssertionError: If cocoGt is None.
        """
        self.iouType = iouType
        self.min_score = min_score
        self.iou_tresh = iou_tresh
        self.useCats = useCats
        self.recall_count = recall_count
        self.cocoGt = copy.deepcopy(cocoGt)
        self.cocoDt = copy.deepcopy(cocoDt)
        self.eval = None

        if iouType == "keypoints":
            self.useCats = True
            self.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
        else:
            self.kpt_oks_sigmas = None

        assert self.cocoGt is not None, "cocoGt is empty"

        if (self.cocoGt is not None) and (self.cocoDt is not None):
            self.drop_cocodt_by_score(min_score=min_score)
            self.evaluate()

    def evaluate(self):
        """Runs COCO evaluation and accumulates results.

        Raises:
            AssertionError: If cocoDt is None.
        """
        assert self.cocoDt is not None, "cocoDt is empty"

        cocoEval = COCOeval_faster(
            self.cocoGt,
            self.cocoDt,
            self.iouType,
            extra_calc=True,
            kpt_oks_sigmas=self.kpt_oks_sigmas,
        )
        cocoEval.params.maxDets = [len(self.cocoGt.anns)]

        self.recThrs = np.linspace(0, 1, self.recall_count + 1, endpoint=True)
        cocoEval.params.recThrs = self.recThrs

        if self.iouType != "keypoints":
            cocoEval.params.iouThrs = [self.iou_tresh]

        cocoEval.params.areaRng = [[0, 10000000000]]
        cocoEval.params.useCats = int(self.useCats)

        self.cocoEval = cocoEval

        cocoEval.evaluate()
        cocoEval.accumulate()

        self.eval = cocoEval.eval

    def drop_cocodt_by_score(self, min_score: float):
        """Removes detection annotations with score below min_score from
        cocoDt.

        Args:
            min_score (float): Minimum score threshold for detections.

        Raises:
            AssertionError: If cocoDt is None.
        """
        assert self.cocoDt is not None, "cocoDt is empty"

        if min_score > 0:
            bad_keys = {}
            bad_images_keys = []

            for key, ann in self.cocoDt.anns.items():
                if ann["score"] < min_score:
                    if bad_keys.get(ann["image_id"]) is None:
                        bad_keys[ann["image_id"]] = {}

                    bad_keys[ann["image_id"]][key] = True

                    bad_images_keys.append(ann["image_id"])

            for image_id in set(bad_images_keys):
                self.cocoDt.imgToAnns[image_id] = [
                    ann for ann in self.cocoDt.imgToAnns[image_id] if bad_keys.get(image_id, {}).get(ann["id"]) is None
                ]

                for ann_id in bad_keys.get(image_id, {}).keys():
                    del self.cocoDt.anns[ann_id]

    @property
    def fp_image_ann_map(self) -> Dict[int, Set[int]]:
        """Gets a mapping from image IDs to sets of annotation IDs for false
        positives.

        Returns:
            Dict[int, Set[int]]: Mapping from image_id to set of annotation IDs marked as false positives.
        """
        image_ann_map = defaultdict(set)
        for ann_id, ann in self.cocoDt.anns.items():
            if ann.get("fp"):
                image_ann_map[ann["image_id"]].add(ann_id)
        return image_ann_map

    @property
    def fn_image_ann_map(self) -> Dict[int, Set[int]]:
        """Gets a mapping from image IDs to sets of annotation IDs for false
        negatives.

        Returns:
            Dict[int, Set[int]]: Mapping from image_id to set of annotation IDs marked as false negatives.
        """
        image_ann_map = defaultdict(set)
        for ann_id, ann in self.cocoGt.anns.items():
            if ann.get("fn"):
                image_ann_map[ann["image_id"]].add(ann_id)
        return image_ann_map
