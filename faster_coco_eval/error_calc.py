from .curves import Curves
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm
import os.path as osp

try:
    import plotly.express as px
    plotly_available = True
except:
    plotly_available = False


logger = logging.getLogger(__name__)


class ErrorCalc(Curves):
    A = 128
    DT_COLOR = (238, 130, 238, A)
    GT_COLOR = (0, 255, 0,   A)

    FN_COLOR = (0, 0, 255,   A)
    FP_COLOR = (255, 0, 0,   A)

    def plot_img(self, img, force_matplot=False, figsize=None, slider=False):
        if plotly_available and not force_matplot:
            if not slider:
                fig = px.imshow(img)
            else:
                fig = px.imshow(img, animation_frame=0,
                                labels=dict(animation_frame="enum"))

            fig.update_layout(coloraxis_showscale=False)
            fig.update_layout(height=600, width=1200)
            # fig.update_xaxes(showticklabels=False)
            # fig.update_yaxes(showticklabels=False)
            fig.show()
        else:
            if figsize is not None:
                plt.figure(figsize=figsize)
            plt.imshow(img, interpolation='nearest')
            plt.axis('off')
            plt.show()

    def print_colors_info(self, _print=False):
        _print_func = logger.info
        if _print:
            _print_func = print

        if logger.getEffectiveLevel() <= 20 or _print:
            _print_func(f"DT_COLOR : {self.DT_COLOR}")
            im = Image.new("RGBA", (64, 32), self.DT_COLOR)
            self.plot_img(im, force_matplot=True, figsize=(1, 0.5))
            _print_func("")

            _print_func(f"GT_COLOR : {self.GT_COLOR}")
            im = Image.new("RGBA", (64, 32), self.GT_COLOR)
            self.plot_img(im, force_matplot=True, figsize=(1, 0.5))
            _print_func("")

            _print_func(f"FN_COLOR : {self.FN_COLOR}")
            im = Image.new("RGBA", (64, 32), self.FN_COLOR)
            self.plot_img(im, force_matplot=True, figsize=(1, 0.5))
            _print_func("")

            _print_func(f"FP_COLOR : {self.FP_COLOR}")
            im = Image.new("RGBA", (64, 32), self.FP_COLOR)
            self.plot_img(im, force_matplot=True, figsize=(1, 0.5))
            _print_func("")

    def _compute_confusion_matrix(self, y_true, y_pred, y_gt_all):
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

    def confusion_matrix(self, _match_results):
        confusion_matrix = self._compute_confusion_matrix([], [], [])
        for _category_id, _match in _match_results.items():
            for image_id in list(_match['fn_list']):
                y_true_dataset = {ann['id']: ann.get(
                    'category_id') for ann in self.dataset['annotations'][image_id]}
                y_pred_dataset = {ann['id']: ann.get(
                    'category_id') for ann in self.result_annotations[image_id]}

                matrix = np.array([[y_true_dataset[row['gt']], y_pred_dataset[row['dt']]]
                                  for row in _match['tp_list'][image_id]])
                y_true = matrix[:, 0].ravel()
                y_pred = matrix[:, 0].ravel()

                confusion_matrix += self._compute_confusion_matrix(
                    y_true, y_pred, list(y_true_dataset.values()))
        return confusion_matrix

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

    def draw_ann(self, draw, ann, color, width=5):
        if self.iouType == 'bbox':
            x1, y1, w, h = ann['bbox']
            draw.rectangle([x1, y1, x1+w, y1+h], outline=color, width=width)
        else:
            for poly in ann['segmentation']:
                if len(poly) > 3:
                    draw.polygon(poly, outline=color, width=width)

    def display_tp_fp_fn(self, _match_results,
                         image_ids=['all'],
                         line_width=7,
                         display_fp=True,
                         display_fn=True,
                         display_tp=True,
                         resize_out_image=None,
                         ):
        image_batch = []
        for _category_id, _match in _match_results.items():
            for image_id in _match['fp_list'].keys():
                if (image_id in image_ids) or 'all' in image_ids:
                    y_true_dataset = {
                        ann['id']: ann for ann in self.dataset['annotations'][image_id]}
                    y_pred_dataset = {
                        ann['id']: ann for ann in self.result_annotations[image_id]}

                    image = self.dataset['images'][image_id]
                    logger.info(f"{image=}")
                    if osp.exists(image.get('file_name')):
                        im = Image.open(image.get('file_name')).convert('RGB')
                    else:
                        logger.warning(
                            f"image {image.get('file_name')} not found in space. load zeros ({image['width']}x{image['height']})")
                        im = Image.new(mode="RGB", size=(
                            image['width'], image['height']))

                    mask = Image.new("RGBA", im.size, (0, 0, 0, 0))
                    draw = ImageDraw.Draw(mask)

                    if display_fp:
                        for fp_ann in _match['fp_list'][image_id]:
                            ann = y_pred_dataset[fp_ann]
                            self.draw_ann(
                                draw, ann, color=self.FP_COLOR, width=line_width)

                    if display_fn:
                        for fp_ann in _match['fn_list'][image_id]:
                            ann = y_true_dataset[fp_ann]
                            self.draw_ann(
                                draw, ann, color=self.FN_COLOR, width=line_width)

                    if display_tp:
                        for row in _match['tp_list'][image_id]:
                            gt_id = row['gt']
                            dt_id = row['dt']

                            ann = y_true_dataset[gt_id]
                            self.draw_ann(
                                draw, ann, color=self.GT_COLOR, width=line_width)

                            ann = y_pred_dataset[dt_id]
                            self.draw_ann(
                                draw, ann, color=self.DT_COLOR, width=line_width)

                    im.paste(mask, mask)
                    if plotly_available:
                        image_batch.append(im)
                    else:
                        self.plot_img(im)

        if len(image_batch) >= 1 and resize_out_image is None:
            resize_out_image = image_batch[0].size

        if len(image_batch) == 1:
            self.plot_img(image_batch[0].resize(resize_out_image))
        elif len(image_batch) > 1:
            image_batch = np.array([np.array(image.resize(resize_out_image))[
                                   :, :, ::-1] for image in image_batch])
            self.plot_img(image_batch, slider=True)
