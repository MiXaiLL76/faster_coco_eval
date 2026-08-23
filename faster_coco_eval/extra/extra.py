import copy
import logging
from collections import defaultdict

import numpy as np

from ..core import COCO, COCOeval_faster

logger = logging.getLogger(__name__)


class ExtraEval:
    """Evaluate COCO results with eager construction-time execution.

    ``iou_tresh`` retains its historical misspelling for API compatibility.
    When both datasets are supplied, construction filters detections and runs
    evaluation immediately; otherwise callers can invoke :meth:`evaluate`.
    """

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
            iou_tresh (float, optional): Historical IoU-threshold parameter
                spelling, retained for compatibility. Defaults to 0.0.
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
            self.kpt_oks_sigmas = None if kpt_oks_sigmas is None else np.array(kpt_oks_sigmas)
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
        if not self.cocoGt.anns:
            logger.warning("Ground-truth annotations are empty; detections will be scored as false positives")
        cocoEval.params.maxDets = [max(1000, len(self.cocoDt.anns))]

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
            bad_ann_ids = set()

            for ann_id, ann in self.cocoDt.anns.items():
                if ann["score"] < min_score:
                    bad_ann_ids.add(ann_id)

            if bad_ann_ids:
                self.cocoDt.dataset["annotations"] = [
                    ann for ann in self.cocoDt.dataset["annotations"] if ann["id"] not in bad_ann_ids
                ]
                self.cocoDt.createIndex()

    @property
    def fp_image_ann_map(self) -> dict[int, set[int]]:
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
    def fn_image_ann_map(self) -> dict[int, set[int]]:
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
