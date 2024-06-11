import copy
import logging

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
        """
        :param cocoGt (COCO):
        :param cocoDt (COCO):
        :param iouType (str): bbox, segm, keypoints
        :param min_score (float):
        :param iou_tresh (float):
        :param recall_count (int):
        :param useCats (bool):
        :param kpt_oks_sigmas (list):
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
        cocoEval.params.useCats = int(self.useCats)  # Выключение labels

        self.cocoEval = cocoEval

        cocoEval.evaluate()
        cocoEval.accumulate()

        self.eval = cocoEval.eval

    def drop_cocodt_by_score(self, min_score: float):
        """
        :param min_score:
        :return:
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
                    ann
                    for ann in self.cocoDt.imgToAnns[image_id]
                    if bad_keys.get(image_id, {}).get(ann["id"]) is None
                ]

                for ann_id in bad_keys.get(image_id, {}).keys():
                    del self.cocoDt.anns[ann_id]
