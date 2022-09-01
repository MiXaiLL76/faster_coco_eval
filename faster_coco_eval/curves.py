import pycocotools._mask as maskUtils
import numpy as np
import json
import time
import logging
from tqdm import tqdm

import matplotlib.pyplot as plt

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    plotly_available = True
except:
    plotly_available = False

logger = logging.getLogger(__name__)


class Curves():
    def __init__(self, annotation_file, iouType='bbox', min_score=0, iou_tresh=0.0, use_native_iou_calc=False):
        self.iouType = iouType
        self.min_score = min_score
        self.iou_tresh = iou_tresh
        self.use_native_iou_calc = use_native_iou_calc

        assert self.iouType in [
            'bbox', 'segm'], f"unknown {iouType} for iou computation"

        logger.debug('loading annotations into memory...')
        tic = time.time()

        dataset = self.load_coco(annotation_file)

        self.dataset = {
            'annotations': {},
            'images': {image['id']: image for image in dataset['images']},
            'categories': dataset['categories'],
        }

        remap_id = False

        dataset_ids = [ann.get('id') for ann in dataset['annotations']]
        unique_count = len(set(dataset_ids))
        if unique_count == len(dataset['annotations']):
            ann_id = len(dataset) + 1
        else:
            ann_id = 1
            logger.warning(
                'dataset have not unique annotation ids. remaping...')
            remap_id = True

        for ann in dataset['annotations']:
            if self.dataset['annotations'].get(ann['image_id']) is None:
                self.dataset['annotations'][ann['image_id']] = []

            if self.iouType == 'segm':
                ann = self.annToRLE(ann)

            if ann.get('id') is None or remap_id:
                ann['id'] = ann_id
                ann_id += 1

            self.dataset['annotations'][ann['image_id']].append(ann)

    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        if ann.get('rle') is not None:
            return ann

        t = self.dataset['images'][ann['image_id']]
        h, w = t['height'], t['width']
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            try:
                rles = maskUtils.frPyObjects(segm, h, w)
            except:
                logger.error(f"{ann=}")
                raise
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            rle = ann['segmentation']

        ann['rle'] = rle

        return ann

    def load_result(self, result_annotations):
        dataset = self.load_coco(result_annotations)

        self.result_annotations = {}

        remap_id = False

        dataset_ids = [ann.get('id') for ann in dataset]
        unique_count = len(set(dataset_ids))
        if unique_count == len(dataset):
            ann_id = len(dataset) + 1
        else:
            ann_id = 1
            logger.warning(
                'loaded result have not unique annotation ids. remaping...')
            remap_id = True

        for ann in dataset:
            if self.result_annotations.get(ann['image_id']) is None:
                self.result_annotations[ann['image_id']] = []

            if ann.get('id') is None or remap_id:
                ann['id'] = ann_id
                ann_id += 1

            if self.iouType == 'segm':
                # Поиск пустых сегментаций. не должно быть так, чтобы они были.
                if np.array(ann['segmentation']).ravel().shape[0] == 0:
                    continue

                ann = self.annToRLE(ann)

            self.result_annotations[ann['image_id']].append(ann)

    def load_coco(self, coco_data):
        if type(coco_data) is str:
            with open(coco_data, 'r') as io:
                dataset = json.load(io)
        elif type(coco_data) in [dict, list]:
            dataset = coco_data
        else:
            dataset = None

        assert (type(dataset) is dict) or ((type(dataset) is list) and (len(dataset) > 0) and (
            type(dataset[0]) is dict)), 'annotation file format not supported'

        return dataset

    def IoU(self, box, boxes):
        """Compute IoU between detect box and gt boxes
        Parameters:
        ----------
        box: numpy array , shape (5, ): x1, y1, x2, y2, score
            input box
        boxes: numpy array, shape (n, 4): x1, y1, x2, y2
            input ground truth boxes
        Returns:
        -------
        ovr: numpy.array, shape (n, )
            IoU
        """
        if len(boxes) == 0:
            logger.error('gt bboxes are not found')
            return np.array([0.])

        box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        area = (boxes[:, 2] - boxes[:, 0] + 1) * \
            (boxes[:, 3] - boxes[:, 1] + 1)

        xx1 = np.maximum(box[0], boxes[:, 0])
        yy1 = np.maximum(box[1], boxes[:, 1])
        xx2 = np.minimum(box[2], boxes[:, 2])
        yy2 = np.minimum(box[3], boxes[:, 3])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (box_area + area - inter)
        return ovr

    def slow_computeIoU(self, gt, dt):
        if self.iouType == 'bbox':
            gt_bboxes = np.float32([g['bbox'] for g in gt])
            # xywh to x1y1x2y2
            gt_bboxes[:, 2:4] += gt_bboxes[:, :2] - 1

            det_bboxes = np.float32([d['bbox'] for d in dt])
            # xywh to x1y1x2y2
            det_bboxes[:, 2:4] += det_bboxes[:, :2] - 1

        else:
            raise Exception('unknown self.iouType for iou computation')

        ious = np.zeros((len(det_bboxes), len(gt_bboxes)), dtype=np.float32)

        for i, det_box in enumerate(det_bboxes):
            ious[i, :] = self.IoU(det_box, gt_bboxes)

        return ious

    def computeIoU(self, gt, dt):
        maxDets = len(gt)

        if self.iouType == 'segm':
            g = [g['rle'] for g in gt]
            d = [d['rle'] for d in dt]
        elif self.iouType == 'bbox':
            g = np.array([g['bbox'] for g in gt])
            d = np.array([d['bbox'] for d in dt])
        else:
            raise Exception('unknown self.iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)

        # ious[ious < self.iou_tresh] = 0

        return ious

    def find_pairs(self, iou, scores, dets):
        dt_count, gt_count = iou.shape

        used_gt = {}
        find = []
        for dt_id in range(dt_count):
            best_gt_id = iou[dt_id, :].argmax()

            iou_value = iou[dt_id, best_gt_id]
            score = scores[dt_id]

            tp = (iou_value >= self.iou_tresh) and not used_gt.get(
                best_gt_id, False) and (score >= self.min_score)

            if tp:
                used_gt[best_gt_id] = True

            find.append([best_gt_id, dt_id, iou_value, score, tp])

        fn_list = [gt_id for gt_id in range(
            gt_count) if not used_gt.get(gt_id, False)]

        return np.array(find).reshape(-1, 5), fn_list

    def select_anns(self, anns, image_id=0, category=None):
        anns = [
            dict(ann, **{'score': ann.get('score', 1)})
            for ann in anns if (ann['image_id'] == image_id) and ((ann['category_id'] == category) or (category is None))]

        anns.sort(key=lambda x: x.get('score'), reverse=True)
        return anns

    def match(self, categories_id=['all']):
        if categories_id is None:
            categories_id = [None]

        assert type(categories_id) is list, f'{categories_id=} not supported'

        categories = {category['id']: category['name']
                      for category in self.dataset['categories']}

        if 'all' in categories_id:
            categories_id = list(categories)

        match_results = {}

        for _image_id in tqdm(self.dataset['images']):
            for _category_id in categories_id:
                if match_results.get(_category_id) is None:
                    match_results[_category_id] = {
                        "num_detectedbox": 0,
                        "num_groundtruthbox": 0,
                        "maxiou_confidence": [],
                        "fn_list": {},
                        "fp_list": {},
                        "tp_list": {},
                    }

                gt_anns = self.dataset['annotations'].get(_image_id, [])
                dt_anns = self.result_annotations.get(_image_id, [])

                gt = self.select_anns(gt_anns, _image_id, _category_id)
                dt = self.select_anns(dt_anns, _image_id, _category_id)

                match_results[_category_id]['num_groundtruthbox'] += len(gt)
                match_results[_category_id]['num_detectedbox'] += len(dt)

                if len(dt) > 0 and len(gt) > 0:
                    if self.use_native_iou_calc:
                        iou = self.slow_computeIoU(gt, dt)
                    else:
                        iou = self.computeIoU(gt, dt)

                    scores = np.float32([d['score'] for d in dt])
                    pairs, fn_list = self.find_pairs(iou, scores, dt)

                    if match_results[_category_id]['fn_list'].get(_image_id) is None:
                        match_results[_category_id]['fn_list'][_image_id] = []
                        match_results[_category_id]['fp_list'][_image_id] = []
                        match_results[_category_id]['tp_list'][_image_id] = []

                    # FN (false negatives), ложноотрицательные – все объекты, присутствующие
                    # в истинной разметке данных, но не предсказанные моделью.
                    match_results[_category_id]['fn_list'][_image_id] += [
                        gt[_ann_i]['id'] for _ann_i in fn_list]

                    # FP (false positives), ложноположительные – все предсказанные объекты,
                    # не являющиеся истинно-положительными;
                    match_results[_category_id]['fp_list'][_image_id] += [
                        dt[int(row[1])]['id'] for row in pairs if not row[4]]

                    # TP (true positives), истинно-положительные – когда предсказанная рамка объекта имеет IoU с \
                    # истинной не ниже порогового значения IoU, а его класс предсказан
                    # с уверенностью не ниже порогового значения уверенности;
                    match_results[_category_id]['tp_list'][_image_id] += [
                        {"dt": dt[int(row[1])]['id'], "gt": gt[int(row[0])]['id']} for row in pairs if row[4]]

                    if pairs.shape[0] > 0:
                        match_results[_category_id]['maxiou_confidence'].append(
                            pairs[:, 2:])

        for _category_id in categories_id:
            match_results[_category_id]['maxiou_confidence'] = np.vstack(
                match_results[_category_id]['maxiou_confidence'])
        return match_results

    def calc_auc(self, recall_list, precision_list):
        # https://towardsdatascience.com/how-to-efficiently-implement-area-under-precision-recall-curve-pr-auc-a85872fd7f14
        mrec = np.concatenate(([0.], recall_list, [1.]))
        mpre = np.concatenate(([0.], precision_list, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    def plot_pre_rec(self, match_results: dict, plotly_backend=True, interp=True):
        use_plotly = False
        if plotly_backend:
            if plotly_available:
                fig = make_subplots(rows=1, cols=1, subplot_titles=[
                                    'Precision-Recall'])
                use_plotly = True
            else:
                logger.warning('plotly not instaled...')

        if not use_plotly:
            fig, axes = plt.subplots(ncols=1)
            fig.set_size_inches(15, 7)
            axes = [axes]

        output = {
            'auc': [],
        }

        for category_id, _match in match_results.items():
            label = _match.get("label", "category_id")
            label = f"[{label}={category_id}] "
            if category_id is None:
                label = ""

            maxiou_confidence = _match['maxiou_confidence']  # iou, score, tp
            num_detectedbox = _match['num_detectedbox']
            num_groundtruthbox = _match['num_groundtruthbox']

            idx = (-maxiou_confidence[:, 1]).argsort(kind='mergesort')

            scores = maxiou_confidence[idx, 1]
            tp = maxiou_confidence[idx, 2]
            fp = -1 * (tp - 1)

            tp_list, fp_list = np.cumsum(tp), np.cumsum(fp)

            precision_list = tp_list / (tp_list + fp_list)
            recall_list = tp_list / num_groundtruthbox

            if interp:
                x_line_vals = np.linspace(0, 1, 3000)
                max_idx = np.argmin(np.abs(x_line_vals - np.max(recall_list)))
                x_line_vals = x_line_vals[:max_idx + 1]
                y_vals = np.interp(x_line_vals, recall_list, precision_list)
                score_vals = np.interp(x_line_vals, recall_list, scores)

                scores = score_vals
                recall_list = x_line_vals
                precision_list = y_vals

            auc = round(self.calc_auc(recall_list, precision_list), 4)

            output['auc'].append(auc)

            if use_plotly:
                fig.add_trace(
                    go.Scatter(
                        x=recall_list,
                        y=precision_list,
                        name=f'{label}auc: {auc:.3f}',
                        mode='lines',
                        text=scores,
                        hovertemplate='Pre: %{y:.3f}<br>' +
                        'Rec: %{x:.3f}<br>' +
                        'Score: %{text:.3f}<extra></extra>',
                        showlegend=True,
                    ),
                    row=1, col=1
                )
            else:
                axes[0].set_title('Precision-Recall')
                axes[0].set_xlabel('Recall')
                axes[0].set_ylabel('Precision')
                axes[0].plot(recall_list, precision_list,
                             label=f'{label}auc: {auc:.3f}')
                axes[0].grid(True)
                axes[0].legend()

        if use_plotly:
            fig.layout.yaxis.range = [0, 1.01]
            fig.layout.xaxis.range = [0, 1.01]

            fig.layout.yaxis.title = 'Precision'
            fig.layout.xaxis.title = 'Recall'

            fig.update_layout(height=600, width=1200)
            fig.show()
        else:
            plt.show()

        return output
