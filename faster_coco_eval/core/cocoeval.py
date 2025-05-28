# Original work Copyright (c) Piotr Dollar and Tsung-Yi Lin, 2014.
# Modified work Copyright (c) 2024 MiXaiLL76

import logging
import os
import sys
from collections import defaultdict
from typing import Callable, List, Optional, Union

if sys.version_info >= (3, 8):
    from typing import Literal

    iouTypeT = Literal["segm", "bbox", "keypoints", "keypoints_crowd", "boundary"]
else:
    iouTypeT = str

import numpy as np

import faster_coco_eval.faster_eval_api_cpp as _C
from faster_coco_eval.core import mask as maskUtils
from faster_coco_eval.core.coco import COCO
from faster_coco_eval.version import __author__, __version__

logger = logging.getLogger(__name__)


class COCOeval:
    def __init__(
        self,
        cocoGt: Optional[COCO] = None,
        cocoDt: Optional[COCO] = None,
        iouType: iouTypeT = "segm",
        print_function: Callable = logger.info,
        extra_calc: bool = False,
        kpt_oks_sigmas: Optional[List[float]] = None,
        use_area: Optional[bool] = True,
        lvis_style: bool = False,
        separate_eval: bool = False,
        boundary_dilation_ratio: float = 0.02,
        boundary_cpu_count: int = min(os.cpu_count(), 4),
    ):
        """Initialize CocoEval using coco APIs for gt and dt.

        Args:
            cocoGt (Optional[COCO]): Object with ground truth annotations.
            cocoDt (Optional[COCO]): Object with detection annotations.
            iouType (iouTypeT): Type of the intersection over union, defaults to "segm".
            print_function (Callable): Function to print output, defaults to logger.info.
            extra_calc (bool): Whether to perform extra calculations, defaults to False.
            kpt_oks_sigmas (Optional[List[float]]): List of sigmas for keypoint evaluation, defaults to None.
            use_area (Optional[bool]): If gt annotations (eg. CrowdPose) do not have 'area', set use_area=False.
            lvis_style (bool): Whether to use LVIS style evaluation, defaults to False.
            separate_eval (bool): Whether to perform separate evaluation, defaults to False.
            boundary_dilation_ratio (float): Ratio for boundary dilation, defaults to 0.02.
            boundary_cpu_count (int): Number of CPUs for boundary computation, defaults to min(os.cpu_count(), 4).
        """
        self.cocoGt: COCO = cocoGt  # ground truth COCO API
        self.cocoDt: COCO = cocoDt  # detections COCO API
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results [KxAxI] elements
        self.eval: dict = {}  # accumulated evaluation results
        self.params = Params(iouType=iouType, kpt_sigmas=kpt_oks_sigmas)  # parameters
        self._paramsEval: dict = {}  # parameters for evaluation
        self.stats: list = []  # result summarization
        self.ious: dict = {}  # ious between all gts and dts

        self.extra_calc = extra_calc
        self.matched = False
        self.lvis_style = lvis_style
        self.separate_eval = separate_eval
        self.boundary_dilation_ratio = boundary_dilation_ratio
        self.boundary_cpu_count = boundary_cpu_count
        self.use_area = use_area

        if iouType == "keypoints" and self.lvis_style:
            logger.warning("lvis_style not supported for keypoint evaluation")
            self.lvis_style = False

        if self.cocoGt is not None:
            self.params.imgIds = sorted(self.cocoGt.getImgIds())
            self.params.catIds = sorted(self.cocoGt.getCatIds())

            if iouType == "keypoints":
                self.params.catIds = sorted(list(self.cocoGt.cat_img_map.keys()))

        self._print_function = print_function  # output print function

        if self.cocoDt is not None:
            if self.print_function == print:
                self.cocoDt.print_function = self.print_function

        if self.cocoGt is not None:
            if self.print_function == print:
                self.cocoGt.print_function = self.print_function

        self.dt_dataset = _C.Dataset()
        self.gt_dataset = _C.Dataset()

    @property
    def print_function(self) -> Callable:
        """
        Returns:
            Callable: The function used for printing/logging.
        """
        return self._print_function

    @print_function.setter
    def print_function(self, value: Callable):
        """Set the print function.

        Args:
            value (Callable): The new print function.
        """
        self._print_function = value

    def _prepare(self):
        """Prepare self.gt_dataset and self.dt_dataset for evaluation based on
        params.

        Populates datasets with annotations, computes RLEs and
        boundaries, and applies LVIS filtering if necessary.
        """
        p = self.params

        # cleanup before create new
        self.gt_dataset.clean()
        self.dt_dataset.clean()

        cat_ids = p.catIds if p.catIds else None

        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=cat_ids))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=cat_ids))

        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if "keypoints" in p.iouType:
                gt["ignore"] = (gt.get("num_keypoints", 0) == 0) or gt["ignore"]

        img_pl = defaultdict(set)  # per image list of categories present in image
        img_nl = {}  # per image map of categories not present in image

        if self.lvis_style:
            img_data = self.cocoGt.load_imgs(ids=p.imgIds)
            img_nl = {d["id"]: d.get("neg_category_ids", []) for d in img_data}
            for ann in gts:
                img_pl[ann["image_id"]].add(ann["category_id"])
            self.img_nel = {d["id"]: d.get("not_exhaustive_category_ids", []) for d in img_data}
            self.freq_groups = self._prepare_freq_group()

        img_sizes = defaultdict(tuple)

        def get_img_size_by_id(image_id: int, dataset: COCO) -> tuple:
            """Get image size by image id.

            Args:
                image_id (int): Image ID.
                dataset (COCO): COCO dataset.

            Returns:
                tuple: (height, width)
            """
            if img_sizes.get(image_id) is None:
                t = dataset.imgs[image_id]
                img_sizes[image_id] = t["height"], t["width"]

        for gt in gts:
            if p.compute_rle:
                get_img_size_by_id(gt["image_id"], self.cocoGt)

        maskUtils.calculateRleForAllAnnotations(
            gts,
            img_sizes,
            p.compute_rle,
            p.compute_boundary,
            self.boundary_dilation_ratio,
            self.boundary_cpu_count,
        )

        for gt in gts:
            self.gt_dataset.append(gt["image_id"], gt["category_id"], gt)

        for dt in dts:
            img_id, cat_id = dt["image_id"], dt["category_id"]
            if self.lvis_style:
                if (cat_id not in img_nl.get(img_id, []) and cat_id not in img_pl[img_id]) and self.lvis_style:
                    dt["drop"] = True
                    continue

                dt["lvis_mark"] = dt["category_id"] in self.img_nel[dt["image_id"]]

            if p.compute_rle:
                get_img_size_by_id(dt["image_id"], self.cocoDt)

        maskUtils.calculateRleForAllAnnotations(
            dts,
            img_sizes,
            p.compute_rle,
            p.compute_boundary,
            self.boundary_dilation_ratio,
            self.boundary_cpu_count,
        )

        for dt in dts:
            if not dt.get("drop", False):
                self.dt_dataset.append(dt["image_id"], dt["category_id"], dt)

    def _prepare_freq_group(self) -> list:
        """Prepare frequency group for LVIS evaluation.

        Returns:
            list: Frequency groups, grouping category indices by frequency label.
        """
        p = self.params
        freq_groups = [[] for _ in p.img_count_lbl]
        cat_data = self.cocoGt.load_cats(p.cat_ids)
        for idx, _cat_data in enumerate(cat_data):
            frequency = _cat_data["frequency"]
            freq_groups[p.img_count_lbl.index(frequency)].append(idx)
        return freq_groups

    def computeIoU(self, imgId: int, catId: int) -> Union[List[float], np.ndarray]:
        """Compute IoU between ground truth and detection for a given image and
        category.

        Args:
            imgId (int): Image ID.
            catId (int): Category ID.

        Returns:
            Union[List[float], np.ndarray]: IoUs between gt and dt for the given image and category.
        """
        p = self.params

        gt = self.gt_dataset.get_instances([imgId], [catId] if p.useCats else p.catIds, bool(p.useCats))[0][
            0
        ]  # 1 imgId  1 catId

        dt = self.dt_dataset.get_instances([imgId], [catId] if p.useCats else p.catIds, bool(p.useCats))[0][
            0
        ]  # 1 imgId  1 catId

        if len(gt) == 0 or len(dt) == 0:
            return []

        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        if p.compute_rle:
            g = [g["rle"] for g in gt]
            d = [d["rle"] for d in dt]
        elif p.iouType == "bbox":
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        else:
            ValueError(f"p.iouType must be bbox or segm or boundary. Get {p.iouType}")

        iscrowd = [int(o.get("iscrowd", 0)) for o in gt]
        # compute iou between each dt and gt region
        ious = maskUtils.iou(d, g, iscrowd)

        if p.compute_boundary:
            g_b = [g["boundary"] for g in gt]
            d_b = [d["boundary"] for d in dt]

            # compute iou between each dt and gt region boundary
            boundary_ious = maskUtils.iou(d_b, g_b, iscrowd)

            # combine mask and boundary iou
            boundary_ious = np.array(boundary_ious)
            iscrowd = np.array(iscrowd)
            if len(gt) and len(dt):
                ious[:, iscrowd == 0] = np.minimum(ious[:, iscrowd == 0], boundary_ious[:, iscrowd == 0])
            else:
                ious = np.minimum(ious, boundary_ious)

        return ious

    def computeOks(self, imgId: int, catId: int) -> np.ndarray:
        """Compute OKS between ground truth and detection for a given image and
        category.

        Args:
            imgId (int): Image ID.
            catId (int): Category ID.

        Returns:
            np.ndarray: OKS between gt and dt for the given image and category.
        """
        p = self.params
        # dimention here should be Nxm
        gts = self.gt_dataset.get(imgId, catId)
        dts = self.dt_dataset.get(imgId, catId)

        inds = np.argsort([-d["score"] for d in dts], kind="mergesort")
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0 : p.maxDets[-1]]
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt["keypoints"])
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt["bbox"]
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt["keypoints"])
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros(k)
                    dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)

                if self.use_area:
                    e = (dx**2 + dy**2) / vars / (gt["area"] + np.spacing(1)) / 2
                else:
                    tmparea = gt["bbox"][3] * gt["bbox"][2] * 0.53
                    e = (dx**2 + dy**2) / vars / (tmparea + np.spacing(1)) / 2

                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """Deprecated. Use COCOeval_faster.evaluateImg instead.

        Args:
            imgId: Image ID.
            catId: Category ID.
            aRng: Area range.
            maxDet: Maximum detections.

        Raises:
            DeprecationWarning: Always.
        """
        raise DeprecationWarning("COCOeval.evaluateImg deprecated! Use COCOeval_faster.evaluateImg instead.")

    def accumulate(self, p=None):
        """Deprecated. Use COCOeval_faster.accumulate instead.

        Args:
            p: Unused.

        Raises:
            DeprecationWarning: Always.
        """
        raise DeprecationWarning("COCOeval.accumulate deprecated! Use COCOeval_faster.accumulate instead.")

    def evaluate(self):
        """Deprecated. Use COCOeval_faster.evaluate instead.

        Raises:
            DeprecationWarning: Always.
        """
        raise DeprecationWarning("COCOeval.evaluate deprecated! Use COCOeval_faster.evaluate instead.")

    def _summarize(
        self,
        ap=1,
        iouThr=None,
        areaRng="all",
        maxDets=100,
        freq_group_idx=None,
        catIds=None,
    ):
        """Summarize evaluation results.

        Args:
            ap (int): 1 for average precision, 0 for average recall.
            iouThr (float, optional): Specific IoU threshold.
            areaRng (str): Area range label.
            maxDets (int): Maximum detections.
            freq_group_idx (int, optional): Frequency group index (for LVIS).
            catIds (list, optional): Category IDs to summarize.

        Returns:
            float: Summary metric.
        """
        p = self.params
        iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} {}] = {:0.3f}"

        freq_str = "catIds={:>3s}".format("all") if self.lvis_style else ""

        titleStr = "Average Precision" if ap == 1 else "Average Recall"
        typeStr = "(AP)" if ap == 1 else "(AR)"
        iouStr = f"{p.iouThrs[0]:0.2f}:{p.iouThrs[-1]:0.2f}" if iouThr is None else f"{iouThr:0.2f}"

        if catIds is not None:
            freq_str = f"catIds=={str(catIds):>3s}"

        if self.lvis_style and (freq_group_idx is not None):
            catIds = self.freq_groups[freq_group_idx]
            freq_str = f"catIds={p.imgCountLbl[freq_group_idx]:>3s}"

        aind = p.areaRngLbl.index(areaRng)
        mind = p.maxDets.index(maxDets)

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval["precision"]
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            if catIds is not None:
                s = s[:, :, catIds, aind, mind]
            else:
                s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval["recall"]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            if catIds is not None:
                s = s[:, catIds, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        self.print_function(
            iStr.format(
                titleStr,
                typeStr,
                iouStr,
                areaRng,
                maxDets,
                freq_str,
                mean_s,
            )
        )
        return mean_s

    def summarize(self):
        """Compute and display summary metrics for evaluation results.

        Note:
            This function can *only* be applied on the default parameter setting.
        """

        def _summarizeDets():
            """Summarize detection metrics.

            Returns:
                np.ndarray: Array of summary statistics.
            """
            _count = 17 if self.lvis_style else 14
            stats = np.zeros((_count,))

            stats[0] = self._summarize(1, maxDets=self.params.maxDets[-1])  # AP_all
            stats[1] = self._summarize(1, iouThr=0.5, maxDets=self.params.maxDets[-1])  # AP_50
            stats[2] = self._summarize(1, iouThr=0.75, maxDets=self.params.maxDets[-1])  # AP_75
            stats[3] = self._summarize(1, areaRng="small", maxDets=self.params.maxDets[-1])  # AP_small
            stats[4] = self._summarize(1, areaRng="medium", maxDets=self.params.maxDets[-1])  # AP_medium
            stats[5] = self._summarize(1, areaRng="large", maxDets=self.params.maxDets[-1])  # AP_large

            if self.lvis_style:
                stats[14] = self._summarize(1, maxDets=self.params.maxDets[-1], freq_group_idx=0)  # APr
                stats[15] = self._summarize(1, maxDets=self.params.maxDets[-1], freq_group_idx=1)  # APc
                stats[16] = self._summarize(1, maxDets=self.params.maxDets[-1], freq_group_idx=2)  # APf

            stats[6] = self._summarize(0, maxDets=self.params.maxDets[0])  # AR_first or AR_all
            if len(self.params.maxDets) >= 2:
                stats[7] = self._summarize(0, maxDets=self.params.maxDets[1])  # AR_second
            if len(self.params.maxDets) >= 3:
                stats[8] = self._summarize(0, maxDets=self.params.maxDets[2])  # AR_third

            stats[9] = self._summarize(0, areaRng="small", maxDets=self.params.maxDets[-1])  # AR_small
            stats[10] = self._summarize(0, areaRng="medium", maxDets=self.params.maxDets[-1])  # AR_medium
            stats[11] = self._summarize(0, areaRng="large", maxDets=self.params.maxDets[-1])  # AR_large

            stats[12] = self._summarize(0, iouThr=0.5, maxDets=self.params.maxDets[-1])  # AR_50
            stats[13] = self._summarize(0, iouThr=0.75, maxDets=self.params.maxDets[-1])  # AR_75

            return stats

        def _summarizeKps():
            """Summarize keypoints metrics.

            Returns:
                np.ndarray: Array of summary statistics.
            """
            stats = np.zeros((10,))
            stats[0] = self._summarize(1, maxDets=self.params.maxDets[-1])  # AP_all
            stats[1] = self._summarize(1, maxDets=self.params.maxDets[-1], iouThr=0.5)  # AP_50
            stats[2] = self._summarize(1, maxDets=self.params.maxDets[-1], iouThr=0.75)  # AP_75
            stats[3] = self._summarize(1, maxDets=self.params.maxDets[-1], areaRng="medium")  # AP_medium
            stats[4] = self._summarize(1, maxDets=self.params.maxDets[-1], areaRng="large")  # AP_large
            stats[5] = self._summarize(0, maxDets=self.params.maxDets[-1])  # AR_all
            stats[6] = self._summarize(0, maxDets=self.params.maxDets[-1], iouThr=0.5)  # AR_50
            stats[7] = self._summarize(0, maxDets=self.params.maxDets[-1], iouThr=0.75)  # AR_75
            stats[8] = self._summarize(0, maxDets=self.params.maxDets[-1], areaRng="medium")  # AR_medium
            stats[9] = self._summarize(0, maxDets=self.params.maxDets[-1], areaRng="large")  # AR_large
            return stats

        def _summarizeKps_crowd():
            """Summarize keypoints metrics for crowd setting.

            Returns:
                np.ndarray: Array of summary statistics.
            """
            stats = np.zeros((9,))
            stats[0] = self._summarize(1, maxDets=self.params.maxDets[-1])  # AP_all
            stats[1] = self._summarize(1, maxDets=self.params.maxDets[-1], iouThr=0.5)  # AP_50
            stats[2] = self._summarize(1, maxDets=self.params.maxDets[-1], iouThr=0.75)  # AP_75
            stats[3] = self._summarize(0, maxDets=self.params.maxDets[-1])  # AR_all
            stats[4] = self._summarize(0, maxDets=self.params.maxDets[-1], iouThr=0.5)  # AR_50
            stats[5] = self._summarize(0, maxDets=self.params.maxDets[-1], iouThr=0.75)  # AR_75
            type_result = self.get_type_result(first=0.2, second=0.8)

            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | type={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision"
            typeStr = "(AP)"
            iouStr = f"{p.iouThrs[0]:0.2f}:{p.iouThrs[-1]:0.2f}"
            self.print_function(iStr.format(titleStr, typeStr, iouStr, "easy", self.params.maxDets[-1], type_result[0]))
            self.print_function(
                iStr.format(titleStr, typeStr, iouStr, "medium", self.params.maxDets[-1], type_result[1])
            )
            self.print_function(iStr.format(titleStr, typeStr, iouStr, "hard", self.params.maxDets[-1], type_result[2]))
            stats[6] = type_result[0]  # AP_easy
            stats[7] = type_result[1]  # AP_medium
            stats[8] = type_result[2]  # AP_hard

            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")

        iouType = self.params.iouType

        if iouType in set(["segm", "bbox", "boundary"]):
            summarize = _summarizeDets
        elif iouType == "keypoints":
            summarize = _summarizeKps
        elif iouType == "keypoints_crowd":
            summarize = _summarizeKps_crowd
        else:
            ValueError(f"iouType must be bbox, segm, boundary or keypoints or keypoints_crowd. Get {iouType}")

        self.all_stats = summarize()
        self.stats = self.all_stats[:12]

    def get_type_result(self, first: float = 0.01, second: float = 0.85) -> list:
        """Calculate type results for easy, medium, and hard splits.

        Args:
            first (float): Threshold for 'easy' crowdIndex.
            second (float): Threshold for 'medium' crowdIndex.

        Returns:
            list: List of scores for [easy, medium, hard].
        """
        easy, mid, hard = self.split(self.cocoGt.annotation_file, first, second)
        res = []
        prev_print = self.print_function
        self.print_function = lambda *args, **kwargs: None

        for curr_type in [easy, mid, hard]:
            curr_list = curr_type
            self.params.imgIds = curr_list
            self.evaluate()
            self.accumulate()
            score = self.eval["precision"][:, :, :, 0, :]
            res.append(round(np.mean(score), 4))

        self.print_function = prev_print
        return res

    def split(self, gt_file: str, first: float = 0.01, second: float = 0.85):
        """Split images into easy, medium, hard according to 'crowdIndex'.

        Args:
            gt_file (str): Path to the ground truth file.
            first (float): Threshold for 'easy' crowdIndex.
            second (float): Threshold for 'medium' crowdIndex.

        Returns:
            tuple: Lists of image ids for (easy, medium, hard).
        """
        data = COCO.load_json(gt_file, use_deepcopy=True)
        easy = []
        mid = []
        hard = []
        for item in data["images"]:
            if item["crowdIndex"] < first:
                easy.append(item["id"])
            elif item["crowdIndex"] < second:
                mid.append(item["id"])
            else:
                hard.append(item["id"])
        return easy, mid, hard

    def __str__(self) -> str:
        """
        Returns:
            str: String representation after summarization.
        """
        self.summarize()
        return str(self.__repr__())

    def __repr__(self) -> str:
        """
        Returns:
            str: Representation of the class with author and version info.
        """
        s = self.__class__.__name__ + "() # "
        s += f"__author__='{__author__}'; "
        s += f"__version__='{__version__}';"
        return s


class Params:
    """Params for coco evaluation api."""

    def setDetParams(self):
        """Set parameters for detection evaluation."""
        self.maxDets = [1, 10, 100]
        self.areaRng = [
            [0**2, 1e5**2],
            [0**2, 32**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]
        self.areaRngLbl = ["all", "small", "medium", "large"]

    def setKpParams(self):
        """Set parameters for keypoint evaluation."""
        self.maxDets = [20]
        self.areaRng = [
            [0**2, 1e5**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]
        self.areaRngLbl = ["all", "medium", "large"]

        self.kpt_oks_sigmas = (
            np.array([
                0.26,
                0.25,
                0.25,
                0.35,
                0.35,
                0.79,
                0.79,
                0.72,
                0.72,
                0.62,
                0.62,
                1.07,
                1.07,
                0.87,
                0.87,
                0.89,
                0.89,
            ])
            / 10.0
        )

    def __init__(
        self,
        iouType: iouTypeT = "segm",
        kpt_sigmas: Optional[List[float]] = None,
    ):
        """Initialize Params for COCO evaluation API.

        Args:
            iouType (iouTypeT): Either "segm", "bbox", "boundary", "keypoints", or "keypoints_crowd".
            kpt_sigmas (Optional[List[float]]): List of keypoint sigma values.
        """
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value # noqa: E501
        self.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        self.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
        self.useCats = 1

        if iouType in set(["segm", "bbox", "boundary"]):
            self.setDetParams()
        elif "keypoints" in iouType:
            self.setKpParams()
            if kpt_sigmas is not None:
                self.kpt_oks_sigmas = np.array(kpt_sigmas)
        else:
            raise TypeError("iouType not supported")

        self.compute_rle = iouType in set(["segm", "boundary"])
        self.compute_boundary = iouType == "boundary"

        self.iouType = iouType

        # We bin categories in three bins based how many images of the training
        # set the category is present in.
        # r: Rare    :  < 10
        # c: Common  : >= 10 and < 100
        # f: Frequent: >= 100
        self.imgCountLbl = ["r", "c", "f"]

    @property
    def useSegm(self) -> int:
        """
        Returns:
            int: 1 if iouType is "segm", else 0.
        """
        return int(self.iouType == "segm")

    @useSegm.setter
    def useSegm(self, value: int):
        """Set segmentation mode and issue deprecation warning.

        Args:
            value (int): 1 for segm, 0 for bbox.
        """
        self.iouType = "segm" if value == 1 else "bbox"
        logger.warning("useSegm is deprecated. Please use iouType (string) instead.")

    @property
    def iou_type(self) -> iouTypeT:
        """
        Returns:
            iouTypeT: IOU type.
        """
        return self.iouType

    @property
    def img_ids(self) -> list:
        """
        Returns:
            list: Image IDs.
        """
        return self.imgIds

    @property
    def cat_ids(self) -> list:
        """
        Returns:
            list: Category IDs.
        """
        return self.catIds

    @property
    def iou_thrs(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: IOU thresholds.
        """
        return self.iouThrs

    @property
    def rec_thrs(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Recall thresholds.
        """
        return self.recThrs

    @property
    def max_dets(self) -> list:
        """
        Returns:
            list: Maximum number of detections.
        """
        return self.maxDets

    @property
    def area_rng(self) -> list:
        """
        Returns:
            list: Area ranges.
        """
        return self.areaRng

    @property
    def area_rng_lbl(self) -> list:
        """
        Returns:
            list: Area range labels.
        """
        return self.areaRngLbl

    @property
    def use_cats(self) -> int:
        """
        Returns:
            int: Whether to use categories.
        """
        return self.useCats

    @property
    def img_count_lbl(self) -> list:
        """
        Returns:
            list: Image count frequency labels.
        """
        return self.imgCountLbl
