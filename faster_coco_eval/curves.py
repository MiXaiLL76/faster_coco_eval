import pycocotools._mask as maskUtils
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Curves():
    def __init__(self, annotation_file, iouType='bbox', min_score=0, iou_tresh=0.0):
        self.iouType   = iouType
        self.min_score = min_score
        self.iou_tresh = iou_tresh
        
        assert self.iouType in ['bbox', 'segm'], f"unknown {iouType} for iou computation"
        
        logger.debug('loading annotations into memory...')
        tic = time.time()

        dataset = self.load_coco(annotation_file)

        self.dataset = {
            'annotations' : {},
            'images' : {image['id'] : image for image in dataset['images']},
            'categories' : dataset['categories'],
        }
        for ann in dataset['annotations']:
            if self.dataset['annotations'].get(ann['image_id']) is None:
                self.dataset['annotations'][ann['image_id']] = []
            
            self.dataset['annotations'][ann['image_id']].append(ann)
    
    def load_result(self, result_annotations):
        dataset = self.load_coco(result_annotations)

        self.result_annotations = {}
        ann_id = len(dataset) + 1
        
        for ann in dataset:
            if self.result_annotations.get(ann['image_id']) is None:
                self.result_annotations[ann['image_id']] = []
            
            if ann.get('id') is None:
                ann['id'] = ann_id
                ann_id += 1
            
            self.result_annotations[ann['image_id']].append(ann)
        
    def load_coco(self, coco_data):
        if type(coco_data) is str:
            with open(coco_data, 'r') as io:
                dataset = json.load(io)
        elif type(coco_data) in [dict, list]:
            dataset = coco_data
        else:
            dataset = None
        
        assert (type(dataset) is dict) or ((type(dataset) is list) and (len(dataset) > 0) and (type(dataset[0]) is dict)), 'annotation file format not supported'
        
        return dataset
    
    def computeIoU(self, gt, dt):
        maxDets = len(gt)
        
        scores = np.float16([d['score'] for d in dt if d['score'] >= self.min_score])

        inds = np.argsort(scores, kind='mergesort')
        dt = [dt[i] for i in inds]

        if self.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif self.iouType == 'bbox':
            g = np.array([g['bbox'] for g in gt])
            d = np.array([d['bbox'] for d in dt])
        else:
            raise Exception('unknown self.iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        ious[ious < self.iou_tresh] = 0
        ious[ious == 0] = -1

        return ious, scores[inds], inds
    
    def find_pairs(self, iou, scores):
        used_gt = {}
        find = []
        for dt_id in range(iou.shape[0]):
            gt_id = iou[dt_id, :].argmax()
            sup_dt_id = iou[:, gt_id].argmax()

            if not used_gt.get(gt_id, False):
                if dt_id == sup_dt_id:
                    used_gt[gt_id] = True

                    _iou  = iou[dt_id, gt_id]

                    if _iou >= self.iou_tresh:
                        find.append([gt_id, dt_id, _iou, scores[dt_id]])
                        iou[dt_id, gt_id] = -1
        
        find.sort(key=lambda x: x[3], reverse=True)
        
        return np.array(find).reshape(-1,4)
    
    def select_anns(self, anns, image_id=0, category=None):
        anns = [
            dict(ann, **{'score': ann.get('score', 1)})
        for ann in anns if (ann['image_id'] == image_id) and ((ann['category_id'] == category) or (category is None))]

        anns.sort(key = lambda x : x.get('score'), reverse=True)
        return anns
    
    
    def match(self, categories_id=['all']):
        if categories_id is None:
            categories_id = [None]
        
        assert type(categories_id) is list, f'{categories_id=} not supported'
        
        categories = {category['id'] : category['name'] for category in self.dataset['categories']}
        
        if 'all' in categories_id:
            categories_id = list(categories)
        
        match_results = {}
        
        for _image_id in tqdm(self.dataset['images']):
            for _category_id in categories_id:
                if match_results.get(_category_id) is None:
                    match_results[_category_id] = {
                        "num_detectedbox"    : 0,
                        "num_groundtruthbox" : 0,
                        "maxiou_confidence"  : []
                    }
                
                gt_anns = self.dataset['annotations'].get(_image_id, [])
                dt_anns = self.result_annotations.get(_image_id, [])
                
                gt = self.select_anns(gt_anns, _image_id, _category_id)
                dt = self.select_anns(dt_anns, _image_id, _category_id)
                
                match_results[_category_id]['num_groundtruthbox'] += len(gt)
                match_results[_category_id]['num_detectedbox'] += len(dt)
                
                
                if len(dt) > 0 and len(gt) > 0:
                    iou, scores, _ = self.computeIoU(gt, dt)
                    pairs = self.find_pairs(iou, scores)
                    
                    if pairs.shape[0] > 0:
                        match_results[_category_id]['maxiou_confidence'].append(pairs[:, 2:])

        for _category_id in categories_id:
            match_results[_category_id]['maxiou_confidence'] = np.vstack(match_results[_category_id]['maxiou_confidence'])
            _ids = np.argsort(match_results[_category_id]['maxiou_confidence'][:, 1], kind='mergesort')
                        
            match_results[_category_id]['maxiou_confidence'] = match_results[_category_id]['maxiou_confidence'][_ids][::-1]
            
        return match_results

    def thres(self, maxiou_confidence, threshold = 0.5):
        maxious = maxiou_confidence[:, 0]
        confidences = maxiou_confidence[:, 1]
        true_or_flase = (maxious > threshold)
        tf_confidence = np.array([true_or_flase, confidences])
        tf_confidence = tf_confidence.T
        tf_confidence = tf_confidence[np.argsort(-tf_confidence[:, 1])]
        return tf_confidence
    
    def plot_curve(self, match_results : dict, threshold_iou=0.5, plot_cols=True, plotly_backend=True):
        if plot_cols:
            fig, axes = plt.subplots(ncols=2)
            fig.set_size_inches(15, 7)
        else:
            fig, axes = plt.subplots(nrows=2)
            fig.set_size_inches(15, 14)
        
        if plotly_backend:
            try:
                from plotly.tools import mpl_to_plotly
            except:
                logger.warning('plotly not instaled...')
                plotly_backend = False
        
        for category_id, _match in match_results.items():
            label = _match.get("label", "category_id")
            label = f"[{label}={category_id}] "
            if category_id is None:
                label = ""
            
            maxiou_confidence  = _match['maxiou_confidence']
            num_detectedbox    = _match['num_detectedbox']
            num_groundtruthbox = _match['num_groundtruthbox']
            
            tf_confidence = self.thres(maxiou_confidence, threshold_iou)

            fp_list = []
            recall_list = []
            precision_list = []
            auc = 0
            mAP = 0
            for num in range(len(tf_confidence)):
                arr = tf_confidence[:(num + 1), 0] # 截取, 注意要加1
                tp = np.sum(arr)
                fp = np.sum(arr == 0)
                recall = tp / num_groundtruthbox
                precision = tp / (tp + fp)
                auc = auc + recall
                mAP = mAP + precision

                fp_list.append(fp)
                recall_list.append(recall)
                precision_list.append(precision)

            auc = auc / len(fp_list)
            mAP = mAP * max(recall_list) / len(recall_list)

            axes[0].set_title('ROC')
            axes[0].set_xlabel('False Positives')
            axes[0].set_ylabel('True Positive rate')
            # plt.ylim(0, 1)
            axes[0].plot(fp_list, recall_list, label = f'{label}AUC: {auc:.3f}')
            axes[0].grid(True)
            axes[0].legend()

            axes[1].set_title('Precision-Recall')
            axes[1].set_xlabel('Recall')
            axes[1].set_ylabel('Precision')
            # plt.axis([0, 1, 0, 1])
            axes[1].plot(recall_list, precision_list, label = f'{label}mAP: {mAP:.3f}')
            axes[1].grid(True)
            axes[1].legend()
        
        
        if plotly_backend:
            pf = mpl_to_plotly(fig, resize=True)
            pf.show()
        else:
            plt.show()