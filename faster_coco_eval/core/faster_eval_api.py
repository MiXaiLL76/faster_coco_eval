# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) 2021 Sartorius AG

import copy
import logging
import numpy as np
import time

from .cocoeval import COCOeval
from . import mask as maskUtils
import faster_coco_eval.faster_eval_api_cpp as _C

logger = logging.getLogger(__name__)


class COCOeval_faster(COCOeval):
    """
    This is a slightly modified version of the original COCO API, where the functions evaluateImg()
    and accumulate() are implemented in C++ to speedup evaluation
    """

    def evaluate(self):
        """
        Run per image evaluation on given images and store results in self.evalImgs_cpp, a
        datastructure that isn't readable from Python but is used by a c++ implementation of
        accumulate().  Unlike the original COCO PythonAPI, we don't populate the datastructure
        self.evalImgs because this datastructure is a computational bottleneck.
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
            (imgId, catId): computeIoU(imgId, catId) for imgId in p.imgIds for catId in catIds
        }  # bottleneck

        maxDet = p.maxDets[-1]

        # <<<< Beginning of code differences with original COCO API
        def convert_instances_to_cpp(instances, is_det=False):
            # Convert annotations for a list of instances in an image to a format that's fast
            # to access in C++
            instances_cpp = []
            for instance in instances:
                instance_cpp = _C.InstanceAnnotation(
                    int(instance["id"]),
                    instance["score"] if is_det else instance.get(
                        "score", 0.0),
                    instance["area"],
                    bool(instance.get("iscrowd", 0)),
                    bool(instance.get("ignore", 0)),
                )
                instances_cpp.append(instance_cpp)
            return instances_cpp

        # Convert GT annotations, detections, and IOUs to a format that's fast to access in C++
        ground_truth_instances = [
            [convert_instances_to_cpp(self._gts[imgId, catId])
             for catId in p.catIds]
            for imgId in p.imgIds
        ]
        detected_instances = [
            [convert_instances_to_cpp(
                self._dts[imgId, catId], is_det=True) for catId in p.catIds]
            for imgId in p.imgIds
        ]
        ious = [[self.ious[imgId, catId] for catId in catIds]
                for imgId in p.imgIds]

        if not p.useCats:
            # For each image, flatten per-category lists into a single list
            ground_truth_instances = [[[o for c in i for o in c]]
                                      for i in ground_truth_instances]
            detected_instances = [[[o for c in i for o in c]]
                                  for i in detected_instances]

        # Call C++ implementation of self.evaluateImgs()
        self._evalImgs_cpp = _C.COCOevalEvaluateImages(
            p.areaRng, maxDet, p.iouThrs, ious, ground_truth_instances, detected_instances
        )
        self._evalImgs = None

        self._paramsEval = copy.deepcopy(self.params)

        toc = time.time()

        self.print_function('COCOeval_opt.evaluate() finished...')
        self.print_function('DONE (t={:0.2f}s).'.format(toc-tic))

    def accumulate(self):
        """
        Accumulate per image evaluation results and store the result in self.eval.  Does not
        support changing parameter settings from those used by self.evaluate()
        """
        self.print_function("Accumulating evaluation results...")
        tic = time.time()
        assert hasattr(
            self, "_evalImgs_cpp"
        ), "evaluate() must be called before accmulate() is called."

        self.eval = _C.COCOevalAccumulate(self._paramsEval, self._evalImgs_cpp)

        # recall is num_iou_thresholds X num_categories X num_area_ranges X num_max_detections
        self.eval["recall"] = np.array(self.eval["recall"]).reshape(
            self.eval["counts"][:1] + self.eval["counts"][2:]
        )

        # precision and scores are num_iou_thresholds X num_recall_thresholds X num_categories X
        # num_area_ranges X num_max_detections
        self.eval["precision"] = np.array(
            self.eval["precision"]).reshape(self.eval["counts"])
        self.eval["scores"] = np.array(
            self.eval["scores"]).reshape(self.eval["counts"])
        
        cat_count = self.eval['counts'][2]
        iou_tresh = self.eval['counts'][0]
        area_ranges = self.eval['counts'][3]

        try:
            self.ground_truth_shape   = [cat_count, area_ranges, iou_tresh, -1]
            self.ground_truth_orig_id = np.array(self.eval['ground_truth_orig_id']).reshape(self.ground_truth_shape)
            self.ground_truth_matches = np.array(self.eval['ground_truth_matches']).reshape(self.ground_truth_shape)
            self.math_matches()
            self.matched = True
        except Exception as e:
            logger.error("math_matches error: ", exc_info=True)
            self.matched = False

        toc = time.time()

        self.print_function('COCOeval_opt.accumulate() finished...')
        self.print_function('DONE (t={:0.2f}s).'.format(toc-tic))
        
        

    def math_matches(self):
        for category_id in range(self.ground_truth_shape[0]):
            for area_range_id in range(self.ground_truth_shape[1]):
                for iou_tresh_id in range(self.ground_truth_shape[2]):
                    for _row, gt_id in enumerate(self.ground_truth_orig_id[category_id,area_range_id,iou_tresh_id]):
                        if gt_id == -1:
                            continue

                        dt_id = self.ground_truth_matches[category_id,area_range_id,iou_tresh_id][_row]

                        _gt_ann = self.cocoGt.anns[gt_id]
                        _dt_ann = self.cocoDt.anns[dt_id]

                        if _gt_ann['image_id'] != _dt_ann['image_id']:
                            continue

                        iou = self.computeAnnIoU(_gt_ann, _dt_ann)
                        
                        if not _gt_ann.get('matched', False):
                            _dt_ann['tp'] = True
                            _dt_ann['gt_id'] = gt_id
                            _dt_ann['iou'] = iou

                            _gt_ann['dt_id'] = dt_id
                            _gt_ann['matched'] = True
                        else:
                            # TODO: Непонятно почему не находит. Проверить на тестовых данных
                            _old_dt_ann = self.cocoDt.anns.get(_gt_ann['dt_id'])
                            if _old_dt_ann is None:
                                continue

                            if _old_dt_ann['id'] == _dt_ann['id']:
                                continue
                            else:
                                if (_old_dt_ann.get('iou', self.computeAnnIoU(_gt_ann, _old_dt_ann)) < iou) or (_old_dt_ann['score'] < _dt_ann['score']):
                                    _dt_ann['tp'] = True
                                    _dt_ann['gt_id'] = gt_id
                                    _dt_ann['iou'] = iou
                                    _gt_ann['dt_id'] = dt_id

                                    for key in ['tp', 'gt_id', 'iou']:
                                        if key in _old_dt_ann:
                                            del _old_dt_ann[key]

        for dt_id in self.cocoDt.anns.keys():
            if self.cocoDt.anns[dt_id].get('gt_id') is None:
                self.cocoDt.anns[dt_id]['fp'] = True

        for gt_id in self.cocoGt.anns.keys():
            if self.cocoGt.anns[gt_id].get('matched') is None:
                self.cocoGt.anns[gt_id]['fn'] = True

    def computeAnnIoU(self, gt_ann, dt_ann):
        g = []
        d = []

        if self.params.iouType == 'segm':
            g.append(gt_ann['rle'])
            d.append(dt_ann['rle'])
        elif self.params.iouType == 'bbox':
            g.append(gt_ann['bbox'])
            d.append(dt_ann['bbox'])
        
        return maskUtils.iou(d, g, [0]).max()
    
    def compute_mIoU(self, categories=None):
        g = []
        d = []
        s = []

        for _, dt_ann in self.cocoDt.anns.items():
            if dt_ann.get('tp', False):
                gt_ann = self.cocoGt.anns[dt_ann['gt_id']]
                if categories is None or gt_ann['category_id'] in categories:
                    s.append(dt_ann.get('score', 1))
                    if self.params.iouType == 'segm':
                        g.append(gt_ann['rle'])
                        d.append(dt_ann['rle'])
                    elif self.params.iouType == 'bbox':
                        g.append(gt_ann['bbox'])
                        d.append(dt_ann['bbox'])
                    else:
                        raise Exception('unknown iouType for iou computation')

        iscrowd = [0 for o in g]
        
        ious = maskUtils.iou(d, g, iscrowd)
        if len(ious) == 0:
            return 0
        else:
            ious = ious.diagonal()
            return ious.mean()
    
    def compute_mAUC(self):
        aucs = []

        for K in range(self.eval['counts'][2]):
            for A in range(self.eval['counts'][3]):            
                precision_list = self.eval['precision'][0, :, K, A, :].ravel()
                
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
        iouType = self.params.iouType
        assert (iouType == 'segm' or iouType ==
                'bbox'), f'iouType={iouType} not supported'

        labels = [
            "AP_all", "AP_50", "AP_75",
            "AP_small", "AP_medium", "AP_large",
            "AR_all", "AR_second", "AR_third",
            "AR_small", "AR_medium", "AR_large", "AR_50", "AR_75"]
        
        if self.matched:
            labels += ["mIoU", "mAUC_" + str(int(self.params.iouThrs[0] * 100))]
        
        maxDets = self.params.maxDets
        if len(maxDets) > 1:
            labels[6] = f'AR_{maxDets[0]}'

        if len(maxDets) >= 2:
            labels[7] = f'AR_{maxDets[1]}'

        if len(maxDets) >= 3:
            labels[8] = f'AR_{maxDets[2]}'

        return {_label: float(self.all_stats[i]) for i, _label in enumerate(labels)}


    @staticmethod
    def calc_auc(recall_list, precision_list):
        # https://towardsdatascience.com/how-to-efficiently-implement-area-under-precision-recall-curve-pr-auc-a85872fd7f14
        # mrec = np.concatenate(([0.], recall_list, [1.]))
        # mpre = np.concatenate(([0.], precision_list, [0.]))
        mrec = recall_list
        mpre = precision_list

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])