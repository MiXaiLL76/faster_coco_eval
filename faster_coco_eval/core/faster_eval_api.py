# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) 2024 MiXaiLL76

import copy
import itertools
import logging
import time
from typing import List, Union

import numpy as np

import faster_coco_eval.faster_eval_api_cpp as _C
from faster_coco_eval.core.cocoeval import COCOeval as COCOevalBase

logger = logging.getLogger(__name__)


class COCOeval_faster(COCOevalBase):
    """This is a slightly modified version of the original COCO API, where the
    functions evaluateImg() and accumulate() are implemented in C++ to speedup
    evaluation."""

    def evaluate(self):
        """Run per image evaluation on given images and store results in
        self.evalImgs_cpp, a datastructure that isn't readable from Python but
        is used by a c++ implementation of accumulate().

        Unlike the original COCO PythonAPI, we don't populate the
        datastructure self.evalImgs because this datastructure is a
        computational bottleneck.
        """
        tic = time.time()

        p = self.params

        self.print_function(f"Evaluate annotation type *{p.iouType}*")

        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()  # not more bottleneck!

        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType in set(["segm", "bbox", "boundary"]):
            computeIoU = self.computeIoU
        elif "keypoints" in p.iouType:
            computeIoU = self.computeOks
        else:
            raise ValueError(f"p.iouType must be segm, bbox, boundary or keypoints. Get {p.iouType}")

        self.ious = {
            (imgId, catId): computeIoU(imgId, catId) for (imgId, catId) in itertools.product(p.imgIds, catIds)
        }  # bottleneck

        # Convert GT annotations, detections, and IOUs to a format that's fast to access in C++ # noqa: E501
        ground_truth_instances = self.gt_dataset.get_cpp_instances(p.imgIds, p.catIds, bool(p.useCats))
        detected_instances = self.dt_dataset.get_cpp_instances(p.imgIds, p.catIds, bool(p.useCats))

        # List Comp faster then map of map
        ious = [[self.ious[imgId, catId] for catId in catIds] for imgId in p.imgIds]

        self._paramsEval = copy.deepcopy(self.params)

        if self.separate_eval:
            # Call C++ implementation of self.evaluateImgs()
            self._evalImgs_cpp = _C.COCOevalEvaluateImages(
                p.areaRng,
                p.maxDets[-1],
                p.iouThrs,
                ious,
                ground_truth_instances,
                detected_instances,
            )
        else:
            self.eval = _C.COCOevalEvaluateAccumulate(
                self._paramsEval,
                ious,
                ground_truth_instances,
                detected_instances,
            )

        toc = time.time()

        self.print_function("COCOeval_opt.evaluate() finished...")
        self.print_function(f"DONE (t={toc - tic:0.2f}s).")

    def accumulate(self):
        """Accumulate per image evaluation results and store the result in
        self.eval.

        Does not support changing parameter settings from those used by
        self.evaluate()
        """
        self.print_function("Accumulating evaluation results...")
        tic = time.time()
        if self.separate_eval:
            assert hasattr(self, "_evalImgs_cpp"), "evaluate() must be called before accmulate() is called."
            self.eval = _C.COCOevalAccumulate(self._paramsEval, self._evalImgs_cpp)

        self.matched = False
        if self.extra_calc:
            try:
                self.math_matches()
                self.matched = True
            except Exception as e:
                logger.error(f"{e} math_matches error: ", exc_info=True)

        toc = time.time()

        self.print_function("COCOeval_opt.accumulate() finished...")
        self.print_function(f"DONE (t={toc - tic:0.2f}s).")

    def math_matches(self):
        """Analyze matched detections and ground truths to assign true
        positive, false positive, and false negative flags, and update
        detection and ground truth annotations in-place.

        Returns:
            None
        """
        for dt_gt, iou in self.eval["matched"].items():
            dt_id, gt_id = dt_gt.split("_")

            dt_id = int(dt_id)
            gt_id = int(gt_id)

            _gt_ann = self.cocoGt.anns[gt_id]
            _dt_ann = self.cocoDt.anns[dt_id]

            _dt_ann["tp"] = True
            _dt_ann["gt_id"] = gt_id
            _dt_ann["iou"] = iou
            _gt_ann["dt_id"] = dt_id

        for dt_id in self.cocoDt.anns.keys():
            if self.cocoDt.anns[dt_id].get("gt_id") is None:
                self.cocoDt.anns[dt_id]["fp"] = True

        for gt_id in self.cocoGt.anns.keys():
            if self.cocoGt.anns[gt_id].get("dt_id") is None:
                self.cocoGt.anns[gt_id]["fn"] = True

    def compute_mIoU(self) -> float:
        """Compute the mean Intersection over Union (mIoU) metric.

        Returns:
            float: Mean IoU across all matched detections and ground truths.
        """
        return sum(self.eval["matched"].values()) / len(self.eval["matched"])

    def compute_mAUC(self) -> float:
        """Compute the mean Area Under Curve (mAUC) metric.

        Returns:
            float: Mean AUC across all categories and area ranges.
        """
        aucs = []

        # K - category
        # A - area_range
        for K in range(self.eval["counts"][2]):
            for A in range(self.eval["counts"][3]):
                precision_list = self.eval["precision"][0, :, K, A, :].ravel()

                recall_list = self.params.recThrs
                auc = COCOeval_faster.calc_auc(recall_list, precision_list)

                if auc != -1:
                    aucs.append(auc)

        if len(aucs):
            return sum(aucs) / len(aucs)
        else:
            return 0

    def summarize(self):
        """Summarize and finalize the statistics of the evaluation.

        Returns:
            None
        """
        super().summarize()

        if self.matched:
            self.all_stats = np.append(self.all_stats, self.compute_mIoU())
            self.all_stats = np.append(self.all_stats, self.compute_mAUC())

    def run(self):
        """Wrapper function which runs the evaluation.

        Calls evaluate(), accumulate(), and summarize() in sequence.

        Returns:
            None
        """
        self.evaluate()
        self.accumulate()
        self.summarize()

    @property
    def extended_metrics(self):
        """Computes extended evaluation metrics for object detection results.

        Calculates per-class and overall (macro) metrics such as mean average precision (mAP) at IoU thresholds,
        precision, recall, and F1-score. Results are computed using evaluation results stored in the object.
        For each class, if categories are used, metrics are reported separately and for the overall dataset.

        Returns:
            dict: A dictionary with the following keys:
                - 'class_map' (list of dict): List of per-class and overall metrics, each as a dictionary containing:
                    - 'class' (str): Class name or "all" for macro metrics.
                    - 'map@50:95' (float): Mean average precision at IoU 0.50:0.95.
                    - 'map@50' (float): Mean average precision at IoU 0.50.
                    - 'precision' (float): Macro-averaged precision.
                    - 'recall' (float): Macro-averaged recall.
                - 'map' (float): Overall mean average precision at IoU 0.50.
                - 'precision' (float): Macro-averaged precision for the best F1-score.
                - 'recall' (float): Macro-averaged recall for the best F1-score.

        Notes:
            - Uses COCO-style evaluation results (precision and scores arrays).
            - Filters out classes with NaN results in any metric.
            - The best F1-score across recall thresholds is used to select macro precision and recall.
        """
        # Extract IoU and recall thresholds from parameters
        iou_thrs, rec_thrs = self.params.iouThrs, self.params.recThrs

        # Indices for IoU=0.50, first area, and last max dets
        iou50_idx, area_idx, maxdet_idx = (int(np.argwhere(np.isclose(iou_thrs, 0.50))), 0, -1)
        P = self.eval["precision"]
        S = self.eval["scores"]

        # Get precision for IoU=0.50, area, and max dets
        prec_raw = P[iou50_idx, :, :, area_idx, maxdet_idx]
        prec = prec_raw.copy().astype(float)
        prec[prec < 0] = np.nan

        # Compute F1 score for each class and recall threshold
        f1_cls = 2 * prec * rec_thrs[:, None] / (prec + rec_thrs[:, None])
        f1_macro = np.nanmean(f1_cls, axis=1)
        best_j = int(f1_macro.argmax())

        # Macro precision and recall at the best F1 score
        macro_precision = float(np.nanmean(prec[best_j]))
        macro_recall = float(rec_thrs[best_j])

        # Score vector for the best recall threshold
        score_vec = S[iou50_idx, best_j, :, area_idx, maxdet_idx].astype(float)
        score_vec[prec_raw[best_j] < 0] = np.nan

        per_class = []
        if self.params.useCats:
            # Map category IDs to names
            cat_ids = self.params.catIds
            cat_id_to_name = {c["id"]: c["name"] for c in self.cocoGt.loadCats(cat_ids)}
            for k, cid in enumerate(cat_ids):
                # Precision per category
                p_slice = P[:, :, k, area_idx, maxdet_idx]
                valid = p_slice > -1
                ap_50_95 = float(p_slice[valid].mean()) if valid.any() else float("nan")
                ap_50 = (
                    float(p_slice[iou50_idx][p_slice[iou50_idx] > -1].mean())
                    if (p_slice[iou50_idx] > -1).any()
                    else float("nan")
                )

                pc = float(prec[best_j, k]) if prec_raw[best_j, k] > -1 else float("nan")
                rc = macro_recall

                # Filter out dataset class if any metric is NaN
                if np.isnan(ap_50_95) or np.isnan(ap_50) or np.isnan(pc) or np.isnan(rc):
                    continue

                per_class.append({
                    "class": cat_id_to_name[int(cid)],
                    "map@50:95": ap_50_95,
                    "map@50": ap_50,
                    "precision": pc,
                    "recall": rc,
                })

        # Add metrics for all classes combined
        per_class.append({
            "class": "all",
            "map@50:95": self.stats_as_dict["AP_all"],
            "map@50": self.stats_as_dict["AP_50"],
            "precision": macro_precision,
            "recall": macro_recall,
        })

        return {
            "class_map": per_class,
            "map": self.stats_as_dict["AP_50"],
            "precision": macro_precision,
            "recall": macro_recall,
        }

    @property
    def stats_as_dict(self):
        """Return the evaluation statistics as a dictionary with descriptive
        labels.

        Returns:
            dict[str, float]: Dictionary mapping metric names to their values.
        """
        if self.params.iouType in set(["segm", "bbox", "boundary"]):
            labels = [
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
                "AR_50",
                "AR_75",
            ]

            if self.lvis_style:
                labels += ["APr", "APc", "APf"]

        elif self.params.iouType == "keypoints":
            labels = [
                "AP_all",
                "AP_50",
                "AP_75",
                "AP_medium",
                "AP_large",
                "AR_all",
                "AR_50",
                "AR_75",
                "AR_medium",
                "AR_large",
            ]
        elif self.params.iouType == "keypoints_crowd":
            labels = [
                "AP_all",
                "AP_50",
                "AP_75",
                "AR_all",
                "AR_50",
                "AR_75",
                "AP_easy",
                "AP_medium",
                "AP_hard",
            ]
        else:
            ValueError(f"iouType must be bbox, segm, boundary or keypoints. Get {self.params.iouType}")

        if self.matched:
            labels += [
                "mIoU",
                "mAUC_" + str(int(self.params.iouThrs[0] * 100)),
            ]

        return {_label: float(self.all_stats[i]) for i, _label in enumerate(labels)}

    @staticmethod
    def calc_auc(
        recall_list: Union[List[float], np.ndarray],
        precision_list: Union[List[float], np.ndarray],
        method: str = "c++",
    ):
        """Calculate area under precision recall curve.

        Args:
            recall_list (Union[List[float], np.ndarray]): List or array of recall values.
            precision_list (Union[List[float], np.ndarray]): List or array of precision values.
            method (str, optional): Method to calculate auc. Defaults to "c++".

        Returns:
            float: Area under precision recall curve.
        """
        # https://towardsdatascience.com/how-to-efficiently-implement-area-under-precision-recall-curve-pr-auc-a85872fd7f14
        if method == "c++":
            return round(_C.calc_auc(recall_list, precision_list), 15)
        else:
            mrec = recall_list
            mpre = precision_list

            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            i = np.where(mrec[1:] != mrec[:-1])[0]
            return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


# Reassignment, for smooth operation of pycocotools replacement
class COCOeval(COCOeval_faster):
    @property
    def print_function(self):
        """Return the print function.

        Returns:
            Callable: The built-in print function.
        """
        return print
