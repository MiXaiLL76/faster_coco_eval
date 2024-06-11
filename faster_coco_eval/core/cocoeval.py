__author__ = "tsungyi"

import copy
import logging
import time
from collections import defaultdict

import numpy as np

from . import mask as maskUtils
from .coco import COCO

logger = logging.getLogger(__name__)


class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(
        self,
        cocoGt=None,
        cocoDt=None,
        iouType="segm",
        print_function=logger.debug,
        extra_calc=False,
        kpt_oks_sigmas=None,
    ):
        """Initialize CocoEval using coco APIs for gt and dt :param cocoGt:

        coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None.

        """
        if not iouType:
            logger.warning("iouType not specified. use default iouType segm")

        self.cocoGt: COCO = cocoGt  # ground truth COCO API
        self.cocoDt: COCO = cocoDt  # detections COCO API
        # per-image per-category evaluation results [KxAxI] elements
        self.evalImgs = defaultdict(list)
        self.eval: dict = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(
            iouType=iouType, kpt_sigmas=kpt_oks_sigmas
        )  # parameters
        self._paramsEval: dict = {}  # parameters for evaluation
        self.stats: list = []  # result summarization
        self.ious: dict = {}  # ious between all gts and dts

        self.extra_calc = extra_calc
        self.matched = False

        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

            if iouType == "keypoints":
                if cocoDt is not None:
                    self.params.catIds = sorted(list(cocoDt.cat_img_map.keys()))
                else:
                    self.params.catIds = sorted(
                        [
                            category_id
                            for category_id, category in cocoGt.cats.items()
                            if len(category.get("keypoints", []))
                        ]
                    )

        self.print_function = print_function  # output print function

    def _prepare(self):
        """Prepare ._gts and ._dts for evaluation based on params.

        :return: None

        """

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann["rle"] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
            dts = self.cocoDt.loadAnns(
                self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == "segm":
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == "keypoints":
                gt["ignore"] = (gt.get("num_keypoints") == 0) or gt["ignore"]
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        # per-image per-category evaluation results
        self.evalImgs = defaultdict(list)
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        """Run per image evaluation on given images and store results (a list
        of dict) in self.evalImgs.

        :return: None

        """
        tic = time.time()
        self.print_function("Running per image evaluation...")
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            logger.warning(
                "useSegm (deprecated) is not None. Running {} evaluation"
                .format(p.iouType)
            )
        self.print_function("Evaluate annotation type *{}*".format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
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
        }

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [
            evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        self.print_function("DONE (t={:0.2f}s).".format(toc - tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        if p.iouType == "segm":
            g = [g["rle"] for g in gt]
            d = [d["rle"] for d in dt]
        elif p.iouType == "bbox":
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        else:
            raise Exception("unknown iouType for iou computation")

        # compute iou between each dt and gt region
        iscrowd = [int(o["iscrowd"]) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d["score"] for d in dts], kind="mergesort")
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0 : p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
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
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + np.max(
                        (z, xd - x1), axis=0
                    )
                    dy = np.max((z, y0 - yd), axis=0) + np.max(
                        (z, yd - y1), axis=0
                    )
                e = (dx**2 + dy**2) / vars / (gt["area"] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """Perform evaluation for single category and image.

        :return: dict (single image results)

        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g["ignore"] or (g["area"] < aRng[0] or g["area"] > aRng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o["iscrowd"]) for o in gt]
        # load computed ious
        ious = (
            self.ious[imgId, catId][:, gtind]
            if len(self.ious[imgId, catId]) > 0
            else self.ious[imgId, catId]
        )

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately # noqa: E501
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]
        # set unmatched detections outside of area range to ignore
        a = np.array(
            [d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]
        ).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
        }

    def accumulate(self, p=None):
        raise DeprecationWarning("deprecated")

    def summarize(self):
        """Compute and display summary metrics for evaluation results.

        Note this functin can *only* be applied on the default parameter
        setting

        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = (
                " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] ="
                " {:0.3f}"
            )
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            self.print_function(
                iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
            )
            return mean_s

        def _summarizeDets():
            stats = np.zeros((14,))
            stats[0] = _summarize(1, maxDets=self.params.maxDets[-1])  # AP_all
            stats[1] = _summarize(
                1, iouThr=0.5, maxDets=self.params.maxDets[-1]
            )  # AP_50
            stats[2] = _summarize(
                1, iouThr=0.75, maxDets=self.params.maxDets[-1]
            )  # AP_75
            stats[3] = _summarize(
                1, areaRng="small", maxDets=self.params.maxDets[-1]
            )  # AP_small
            stats[4] = _summarize(
                1, areaRng="medium", maxDets=self.params.maxDets[-1]
            )  # AP_medium
            stats[5] = _summarize(
                1, areaRng="large", maxDets=self.params.maxDets[-1]
            )  # AP_large

            # AR_first or AR_all
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            if len(self.params.maxDets) >= 2:
                stats[7] = _summarize(
                    0, maxDets=self.params.maxDets[1]
                )  # AR_second
            if len(self.params.maxDets) >= 3:
                stats[8] = _summarize(
                    0, maxDets=self.params.maxDets[2]
                )  # AR_third

            stats[9] = _summarize(
                0, areaRng="small", maxDets=self.params.maxDets[-1]
            )  # AR_small
            stats[10] = _summarize(
                0, areaRng="medium", maxDets=self.params.maxDets[-1]
            )  # AR_medium
            stats[11] = _summarize(
                0, areaRng="large", maxDets=self.params.maxDets[-1]
            )  # AR_large

            stats[12] = _summarize(
                0, iouThr=0.5, maxDets=self.params.maxDets[-1]
            )  # AR_50
            stats[13] = _summarize(
                0, iouThr=0.75, maxDets=self.params.maxDets[-1]
            )  # AR_75
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=self.params.maxDets[-1])
            stats[1] = _summarize(
                1, maxDets=self.params.maxDets[-1], iouThr=0.5
            )
            stats[2] = _summarize(
                1, maxDets=self.params.maxDets[-1], iouThr=0.75
            )
            stats[3] = _summarize(
                1, maxDets=self.params.maxDets[-1], areaRng="medium"
            )
            stats[4] = _summarize(
                1, maxDets=self.params.maxDets[-1], areaRng="large"
            )
            stats[5] = _summarize(0, maxDets=self.params.maxDets[-1])
            stats[6] = _summarize(
                0, maxDets=self.params.maxDets[-1], iouThr=0.5
            )
            stats[7] = _summarize(
                0, maxDets=self.params.maxDets[-1], iouThr=0.75
            )
            stats[8] = _summarize(
                0, maxDets=self.params.maxDets[-1], areaRng="medium"
            )
            stats[9] = _summarize(
                0, maxDets=self.params.maxDets[-1], areaRng="large"
            )
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        elif iouType == "keypoints":
            summarize = _summarizeKps

        self.all_stats = summarize()
        self.stats = self.all_stats[:12]

    def __str__(self):
        self.summarize()


class Params:
    """Params for coco evaluation api."""

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value # noqa: E501
        self.iouThrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        self.maxDets = [1, 10, 100]
        self.areaRng = [
            [0**2, 1e5**2],
            [0**2, 32**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]
        self.areaRngLbl = ["all", "small", "medium", "large"]
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value # noqa: E501
        self.iouThrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        self.maxDets = [20]
        self.areaRng = [
            [0**2, 1e5**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]
        self.areaRngLbl = ["all", "medium", "large"]
        self.useCats = 1

        self.kpt_oks_sigmas = (
            np.array(
                [
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
                ]
            )
            / 10.0
        )

    def __init__(self, iouType="segm", kpt_sigmas=None):
        """Params for coco evaluation api.

        IouType: the type of iou to use for evaluation, can be 'segm', 'bbox',
            or 'keypoints'
        kpt_sigmas: list of keypoint sigma values.

        """
        if iouType == "segm" or iouType == "bbox":
            self.setDetParams()
        elif iouType == "keypoints":
            self.setKpParams()
            if kpt_sigmas is not None:
                self.kpt_oks_sigmas = np.array(kpt_sigmas)
        else:
            raise Exception("iouType not supported")
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
