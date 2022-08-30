from .curves import Curves
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm
import os.path as osp

logger = logging.getLogger(__name__)


class ErrorCalc(Curves):
    def compute_confusion_matrix(self, y_true, y_pred, y_gt_all):
        categories = self.dataset['categories']
        K = len(categories)

        categories_id = {category['id']: _i for _i,
                         category in enumerate(categories)}

        cm = np.zeros((K, K + 1), dtype=np.int32)
        for a, p in zip(y_true, y_pred):
            cm[categories_id[a]][categories_id[p]] += 1

        y_gt_all = np.array(y_gt_all)

        for key, val in categories_id.items():
            cm[val][K] = (y_gt_all == key).sum() - cm[val][:K].sum()

        return cm

    def display_matrix(self, conf_matrix, in_percent=True, figsize=(10, 10), fontsize=16):
        names = [category['name']
                 for category in self.dataset['categories']] + ['not detected']

        if in_percent:
            sum_by_col = conf_matrix.sum(axis=1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.matshow(conf_matrix, cmap='Blues', alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):

                value = conf_matrix[i, j]

                if in_percent:
                    value = int(value / sum_by_col[i] * 100)

                if value > 0:
                    ax.text(x=j, y=i, s=value, va='center', ha='center')

        plt.xlabel('Predictions', fontsize=fontsize)
        plt.ylabel('Actuals', fontsize=fontsize)

        plt.xticks(list(range(len(names))), names, rotation=90)
        plt.yticks(list(range(len(names[:-1]))), names[:-1])

        title = 'Confusion Matrix'
        if in_percent:
            title += ' [%]'

        plt.title(title, fontsize=fontsize)
        plt.show()

    def display_fp(self, compute_result, top=5, margin=50, img_prefix=None):
        fp_anns = []
        for _image_id, annotation_ids in compute_result.items():
            dt_anns = self.result_annotations.get(_image_id, [])
            fp_anns += [ann for ann in dt_anns if ann['id']
                        in compute_result[_image_id]['fp']]

        fp_anns.sort(key=lambda ann: ann['score'], reverse=True)

        for ann in fp_anns[:top]:
            image_fn = self.dataset['images'][ann['image_id']]['file_name']
            if type(img_prefix) is str:
                image_fn = osp.join(img_prefix, image_fn)

            img = Image.open(image_fn)
            img_mask = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img_mask)

            for gt_ann in self.dataset['annotations'].get(ann['image_id'], []):
                if gt_ann['category_id'] == ann['category_id']:
                    if self.iouType == 'segm':
                        for _segm in gt_ann['segmentation']:
                            draw.polygon(_segm, outline=(
                                0, 255, 0, 255), fill=(0, 255, 0, 64))
                    else:
                        x1, y1, w, h = gt_ann['bbox']
                        x2 = x1 + w
                        y2 = y1 + h
                        draw.rectangle([x1, y1, x2, y2], outline=(
                            0, 255, 0, 255), fill=(0, 255, 0, 64))

            x1, y1, w, h = ann['bbox']
            x2 = x1 + w
            y2 = y1 + h

            if self.iouType == 'segm':
                for _segm in ann['segmentation']:
                    draw.polygon(_segm, outline=(255, 0, 0, 255),
                                 fill=(255, 0, 0, 64))
            else:
                draw.rectangle([x1, y1, x2, y2], outline=(
                    255, 0, 0, 255), fill=(255, 0, 0, 64))

            print(image_fn)
            print(ann['bbox'])

            img.paste(img_mask, img_mask)

            plt.imshow(img.crop((x1-margin, y1-margin, x2+margin, y2+margin)))
            plt.show()

    def display_fn(self, compute_result, top=5, margin=50, img_prefix=None):
        fn_anns = []
        for _image_id, annotation_ids in compute_result.items():
            dt_anns = self.result_annotations.get(_image_id, [])
            fn_anns += [ann for ann in dt_anns if ann['id']
                        in compute_result[_image_id]['fn']]

        for ann in fn_anns[:top]:
            image_fn = self.dataset['images'][ann['image_id']]['file_name']
            if type(img_prefix) is str:
                image_fn = osp.join(img_prefix, image_fn)

            img = Image.open(image_fn)
            img_mask = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img_mask)

            for dt_ann in self.result_annotations.get(ann['image_id'], []):
                if dt_ann['category_id'] == ann['category_id']:
                    if self.iouType == 'segm':
                        for _segm in dt_ann['segmentation']:
                            draw.polygon(_segm, outline=(
                                0, 255, 0, 255), fill=(0, 255, 0, 64))
                    else:
                        x1, y1, w, h = dt_ann['bbox']
                        x2 = x1 + w
                        y2 = y1 + h
                        draw.rectangle([x1, y1, x2, y2], outline=(
                            0, 255, 0, 255), fill=(0, 255, 0, 64))

            x1, y1, w, h = ann['bbox']
            x2 = x1 + w
            y2 = y1 + h

            if self.iouType == 'segm':
                for _segm in ann['segmentation']:
                    draw.polygon(_segm, outline=(0, 0, 255, 255),
                                 fill=(0, 0, 255, 64))
            else:
                draw.rectangle([x1, y1, x2, y2], outline=(
                    0, 0, 255, 255), fill=(0, 0, 255, 64))

            print(image_fn)
            print(ann['bbox'])

            img.paste(img_mask, img_mask)

            plt.imshow(img.crop((x1-margin, y1-margin, x2+margin, y2+margin)))
            plt.show()

    def compute_errors(self):
        confusion_matrix = self.compute_confusion_matrix([], [], [])

        result_annotations = {}
        for _image_id in tqdm(self.dataset['images']):
            result_annotations[_image_id] = {
                "tp": [],
                "fn": [],
                "fp": [],
            }

            gt_anns = self.dataset['annotations'].get(_image_id, [])
            dt_anns = self.result_annotations.get(_image_id, [])

            num_groundtruthbox = len(gt_anns)
            num_detectedbox = len(dt_anns)

            if num_groundtruthbox > 0 and num_detectedbox > 0:
                iou = self.computeIoU(gt_anns, dt_anns)

                # GT
                gt_categories = np.int32(
                    [ann['category_id'] for ann in gt_anns])
                gt_real_idx = np.int32([ann['id'] for ann in gt_anns])

                # print(dt_anns)

                # DETECT
                dt_categories = np.int32(
                    [dt_ann['category_id'] for dt_ann in dt_anns])
                dt_real_idx = np.int32([dt_ann['id'] for dt_ann in dt_anns])

                # COMPARE
                scores = np.float16([dt_ann['score'] for dt_ann in dt_anns])
                find = self.find_pairs(iou, scores)

                # GT
                gt_filter_ids = find[:, 0].astype(np.int32)
                filtred_gt_categories = gt_categories[gt_filter_ids]
                filtred_gt_real_idx = gt_real_idx[gt_filter_ids]

                # DETECT
                dt_filter_ids = find[:, 1].astype(np.int32)
                filtred_dt_categories = dt_categories[dt_filter_ids]
                filtred_dt_real_idx = dt_real_idx[dt_filter_ids]

                # TP (true positives), истинно-положительные – когда предсказанная рамка объекта имеет IoU с \
                # истинной не ниже порогового значения IoU, а его класс предсказан
                # с уверенностью не ниже порогового значения уверенности;
                tp_mask = (filtred_gt_categories == filtred_dt_categories)
                tp_ids = tp_mask.nonzero()[0]
                # tp = len(tp_ids)

                result_annotations[_image_id]['tp'] = filtred_gt_real_idx[tp_ids]

                # FN (false negatives), ложноотрицательные – все объекты, присутствующие
                # в истинной разметке данных, но не предсказанные моделью.

                # fn = num_groundtruthbox - tp
                fn_mask = np.in1d(
                    gt_real_idx, result_annotations[_image_id]['tp'], invert=True)
                result_annotations[_image_id]['fn'] = gt_real_idx[fn_mask]

                # FP (false positives), ложноположительные – все предсказанные объекты,
                # не являющиеся истинно-положительными;

                # fp = len(dt_idx) - tp

                tp_det = filtred_dt_real_idx[tp_ids]
                fp_mask = np.in1d(dt_real_idx, tp_det, invert=True)
                result_annotations[_image_id]['fp'] = dt_real_idx[fp_mask]

                image_confusion_matrix = self.compute_confusion_matrix(
                    filtred_gt_categories, filtred_dt_categories, gt_categories)

                confusion_matrix += image_confusion_matrix

        return confusion_matrix, result_annotations
