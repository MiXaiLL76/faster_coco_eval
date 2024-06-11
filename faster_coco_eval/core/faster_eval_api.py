# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) 2021 Sartorius AG

import copy
import logging
import time

import numpy as np

import faster_coco_eval.faster_eval_api_cpp as _C

from .cocoeval import COCOeval

logger = logging.getLogger(__name__)


class COCOeval_faster(COCOeval):
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
        :return: None

        """
        tic = time.time()

        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"

        self.print_function("Evaluate annotation type *{}*".format(p.iouType))

        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()  # bottleneck

        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == "segm" or p.iouType == "bbox":
            computeIoU = self.computeIoU
        elif p.iouType == "keypoints":
            computeIoU = self.computeOks
        self.ious = {
            (imgId, catId): computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds
        }  # bottleneck

        maxDet = p.maxDets[-1]

        # <<<< Beginning of code differences with original COCO API
        def convert_instances_to_cpp(instances, is_det=False):
            # Convert annotations for a list of instances in an image to a format that's fast # noqa: E501
            # to access in C++
            instances_cpp = []
            for instance in instances:
                instance_cpp = _C.InstanceAnnotation(
                    int(instance["id"]),
                    instance["score"] if is_det else instance.get("score", 0.0),
                    instance["area"],
                    bool(instance.get("iscrowd", 0)),
                    bool(instance.get("ignore", 0)),
                )
                instances_cpp.append(instance_cpp)
            return instances_cpp

        # Convert GT annotations, detections, and IOUs to a format that's fast to access in C++ # noqa: E501
        ground_truth_instances = [
            [
                convert_instances_to_cpp(self._gts[imgId, catId])
                for catId in p.catIds
            ]
            for imgId in p.imgIds
        ]
        detected_instances = [
            [
                convert_instances_to_cpp(self._dts[imgId, catId], is_det=True)
                for catId in p.catIds
            ]
            for imgId in p.imgIds
        ]
        ious = [
            [self.ious[imgId, catId] for catId in catIds] for imgId in p.imgIds
        ]

        if not p.useCats:
            # For each image, flatten per-category lists into a single list
            ground_truth_instances = [
                [[o for c in i for o in c]] for i in ground_truth_instances
            ]
            detected_instances = [
                [[o for c in i for o in c]] for i in detected_instances
            ]

        # Call C++ implementation of self.evaluateImgs()
        self._evalImgs_cpp = _C.COCOevalEvaluateImages(
            p.areaRng,
            maxDet,
            p.iouThrs,
            ious,
            ground_truth_instances,
            detected_instances,
        )
        self._evalImgs = None

        self._paramsEval = copy.deepcopy(self.params)

        toc = time.time()

        self.print_function("COCOeval_opt.evaluate() finished...")
        self.print_function("DONE (t={:0.2f}s).".format(toc - tic))

    def accumulate(self):
        """Accumulate per image evaluation results and store the result in
        self.eval.

        Does not support changing parameter settings from those used by
        self.evaluate()

        """
        self.print_function("Accumulating evaluation results...")
        tic = time.time()
        assert hasattr(
            self, "_evalImgs_cpp"
        ), "evaluate() must be called before accmulate() is called."

        self.eval = _C.COCOevalAccumulate(self._paramsEval, self._evalImgs_cpp)

        # recall is num_iou_thresholds X num_categories X num_area_ranges X num_max_detections # noqa: E501
        self.eval["recall"] = np.array(self.eval["recall"]).reshape(
            self.eval["counts"][:1] + self.eval["counts"][2:]
        )

        # precision and scores are num_iou_thresholds X num_recall_thresholds X num_categories X num_area_ranges X num_max_detections # noqa: E501
        self.eval["precision"] = np.array(self.eval["precision"]).reshape(
            self.eval["counts"]
        )
        self.eval["scores"] = np.array(self.eval["scores"]).reshape(
            self.eval["counts"]
        )

        self.matched = False
        try:
            if self.extra_calc:

                num_iou_thresholds, _, _, num_area_ranges, _ = self.eval[
                    "counts"
                ]

                self.detection_matches = np.vstack(
                    np.array(self.eval["detection_matches"]).reshape(
                        num_iou_thresholds, num_area_ranges, -1
                    )
                )
                assert self.detection_matches.shape[1] <= len(self.cocoDt.anns)

                self.ground_truth_matches = np.vstack(
                    np.array(self.eval["ground_truth_matches"]).reshape(
                        num_iou_thresholds, num_area_ranges, -1
                    )
                )
                assert self.ground_truth_matches.shape[1] <= len(
                    self.cocoGt.anns
                )

                self.ground_truth_orig_id = np.vstack(
                    np.array(self.eval["ground_truth_orig_id"]).reshape(
                        num_iou_thresholds, num_area_ranges, -1
                    )
                )
                assert self.ground_truth_orig_id.shape[1] <= len(
                    self.cocoGt.anns
                )
                self.math_matches()
                self.matched = True
        except Exception as e:
            logger.error("{} math_matches error: ".format(e), exc_info=True)

        toc = time.time()

        self.print_function("COCOeval_opt.accumulate() finished...")
        self.print_function("DONE (t={:0.2f}s).".format(toc - tic))

    def math_matches(self):
        """For each ground truth, find the best matching detection and set the
        detection as matched."""
        for gidx, ground_truth_matches in enumerate(self.ground_truth_matches):
            gt_ids = self.ground_truth_orig_id[gidx]

            for idx, dt_id in enumerate(ground_truth_matches):
                if dt_id == 0:
                    continue

                gt_id = gt_ids[idx]
                if gt_id <= -1:
                    continue

                _gt_ann = self.cocoGt.anns[gt_id]
                _dt_ann = self.cocoDt.anns[dt_id]
                _img_id = self.cocoGt.ann_img_map[gt_id]
                _catId = _gt_ann["category_id"] if self.params.useCats else -1

                if self.params.useCats:
                    _catId = _gt_ann["category_id"]
                    _map_gt_dict = self.cocoGt.img_cat_ann_idx_map
                    _map_dt_dict = self.cocoDt.img_cat_ann_idx_map
                    _map_id = (_img_id, _catId)
                else:
                    _catId = -1
                    _map_gt_dict = self.cocoGt.img_ann_idx_map
                    _map_dt_dict = self.cocoDt.img_ann_idx_map
                    _map_id = _img_id

                iou_gt_id = _map_gt_dict[_map_id].get(gt_id)
                iou_dt_id = _map_dt_dict[_map_id].get(dt_id)

                if iou_gt_id is None or iou_dt_id is None:
                    continue

                iou = self.ious[(_img_id, _catId)][iou_dt_id, iou_gt_id]

                if not _gt_ann.get("matched", False):
                    _dt_ann["tp"] = True
                    _dt_ann["gt_id"] = gt_id
                    _dt_ann["iou"] = iou

                    _gt_ann["dt_id"] = dt_id
                    _gt_ann["matched"] = True
                else:
                    _old_dt_ann = self.cocoDt.anns[_gt_ann["dt_id"]]

                    if _old_dt_ann.get("iou", 0) < iou:
                        for _key in ["tp", "gt_id", "iou"]:
                            if _old_dt_ann.get(_key) is not None:
                                del _old_dt_ann[_key]

                        _dt_ann["tp"] = True
                        _dt_ann["gt_id"] = gt_id
                        _dt_ann["iou"] = iou

                        _gt_ann["dt_id"] = dt_id

        for dt_id in self.cocoDt.anns.keys():
            if self.cocoDt.anns[dt_id].get("gt_id") is None:
                self.cocoDt.anns[dt_id]["fp"] = True

        for gt_id in self.cocoGt.anns.keys():
            if self.cocoGt.anns[gt_id].get("matched") is None:
                self.cocoGt.anns[gt_id]["fn"] = True

    def compute_mIoU(self):
        """Compute the mIoU metric."""
        ious = []
        for _, dt_ann in self.cocoDt.anns.items():
            if dt_ann.get("iou", False):
                ious.append(dt_ann["iou"])
        return sum(ious) / len(ious)

    def compute_mAUC(self):
        """Compute the mAUC metric."""
        aucs = []

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
        super().summarize()

        if self.matched:
            self.all_stats = np.append(self.all_stats, self.compute_mIoU())
            self.all_stats = np.append(self.all_stats, self.compute_mAUC())

    @property
    def stats_as_dict(self):
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
        ]

        if self.params.iouType in ["segm", "bbox"]:
            labels += ["AR_50", "AR_75"]
        else:
            labels = [label for label in labels if "small" not in label]

        if self.matched:
            labels += [
                "mIoU",
                "mAUC_" + str(int(self.params.iouThrs[0] * 100)),
            ]

        maxDets = self.params.maxDets
        if len(maxDets) > 1:
            labels[6] = "AR_{}".format(maxDets[0])

        if len(maxDets) >= 2:
            labels[7] = "AR_{}".format(maxDets[1])

        if len(maxDets) >= 3:
            labels[8] = "AR_{}".format(maxDets[2])

        return {
            _label: float(self.all_stats[i]) for i, _label in enumerate(labels)
        }

    @staticmethod
    def calc_auc(recall_list, precision_list):
        """
        Calculate area under precision recall curve
        recall_list: list of recall values
        precision_list: list of precision values
        """
        # https://towardsdatascience.com/how-to-efficiently-implement-area-under-precision-recall-curve-pr-auc-a85872fd7f14
        # mrec = np.concatenate(([0.], recall_list, [1.]))
        # mpre = np.concatenate(([0.], precision_list, [0.]))
        mrec = recall_list
        mpre = precision_list

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
