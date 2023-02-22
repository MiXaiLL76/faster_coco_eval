from ..core.coco import COCO
from ..core.cocoeval import COCOeval
from ..core import mask as maskUtils

from PIL import Image, ImageDraw
import numpy as np
import logging
import os.path as osp

import matplotlib.pyplot as plt

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    plotly_available = True
except:
    plotly_available = False

logger = logging.getLogger(__name__)


class Curves():
    A = 128
    DT_COLOR = (238, 130, 238, A)

    GT_COLOR = (0, 255, 0,   A)
    FN_COLOR = (0, 0, 255,   A)
    FP_COLOR = (255, 0, 0,   A)

    def __init__(self,
                 cocoGt: COCO = None,
                 cocoDt: COCO = None,
                 iouType: str = 'bbox',
                 min_score: float = 0,
                 iou_tresh: float = 0.0,
                 recall_count: int = 100,
                 useCats: bool = False,
                 ):
        self.iouType = iouType
        self.min_score = min_score
        self.iou_tresh = iou_tresh
        self.useCats = useCats
        self.recall_count = recall_count
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt

        self.evaluate()

    def evaluate(self):
        cocoEval = COCOeval(self.cocoGt, self.cocoDt, self.iouType)
        cocoEval.params.maxDets = [len(self.cocoGt.anns)]

        cocoEval.params.iouThr = [0, 0.5]
        cocoEval.params.iouThrs = [self.iou_tresh]
        cocoEval.params.areaRng = [[0, 10000000000]]
        self.recThrs = np.linspace(0, 1, self.recall_count + 1, endpoint=True)
        cocoEval.params.recThrs = self.recThrs

        cocoEval.params.useCats = int(self.useCats)  # Выключение labels

        cocoEval.evaluate()
        cocoEval.accumulate()

        self.eval = cocoEval.eval
        self.math_matches()

    def math_matches(self):
        for gt_id, dt_id, is_tp in self.eval['matches']:
            is_tp = bool(is_tp)

            self.cocoDt.anns[dt_id]['tp'] = is_tp

            if is_tp:
                self.cocoGt.anns[gt_id]['tp'] = is_tp
                self.cocoGt.anns[gt_id]['dt_id'] = dt_id
                #
                self.cocoDt.anns[dt_id]['gt_id'] = gt_id

        for gt_id in self.cocoGt.anns.keys():
            if self.cocoGt.anns[gt_id].get('tp') is None:
                self.cocoGt.anns[gt_id]['fn'] = True

    def remap_categories_on_tp(self):
        # remap categories
        for dt_id, ann in self.cocoDt.anns.items():
            if ann.get('tp', False):
                self.cocoDt.anns[dt_id]['category_id'] = self.cocoGt.anns[ann['gt_id']]['category_id']

        self.useCats = True
        self.evaluate()
        # remap categories

    def calc_auc(self, recall_list, precision_list):
        # https://towardsdatascience.com/how-to-efficiently-implement-area-under-precision-recall-curve-pr-auc-a85872fd7f14
        # mrec = np.concatenate(([0.], recall_list, [1.]))
        # mpre = np.concatenate(([0.], precision_list, [0.]))
        mrec = recall_list
        mpre = precision_list

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    def build_curve(self, label):
        curve = []

        if self.useCats:
            cat_ids = list(range(self.eval['precision'].shape[2]))
        else:
            cat_ids = [0]

        for category_id in cat_ids:
            _label = f"[{label}={category_id}] "
            if len(cat_ids) == 1:
                _label = ""

            precision_list = self.eval['precision'][:,
                                                    :, category_id, :, :].ravel()
            recall_list = self.recThrs
            scores = self.eval['scores'][:, :, category_id, :, :].ravel()
            auc = round(self.calc_auc(recall_list, precision_list), 4)

            curve.append(dict(
                recall_list=recall_list,
                precision_list=precision_list,
                name=f'{_label}auc: {auc:.3f}',
                scores=scores,
                auc=auc,
                category_id=category_id,
            ))

        return curve

    def plot_pre_rec(self, curves=None, plotly_backend=False, label="category_id"):
        if curves is None:
            curves = self.build_curve(label)

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

        for _curve in curves:
            recall_list = _curve['recall_list']
            precision_list = _curve['precision_list']
            scores = _curve['scores']
            name = _curve['name']

            if use_plotly:
                fig.add_trace(
                    go.Scatter(
                        x=recall_list,
                        y=precision_list,
                        name=name,
                        text=scores,
                        hovertemplate='Pre: %{y:.3f}<br>' +
                        'Rec: %{x:.3f}<br>' +
                        'Score: %{text:.3f}<extra></extra>',
                        showlegend=True,
                        mode='lines',
                    ),
                    row=1, col=1
                )
            else:
                axes[0].set_title('Precision-Recall')
                axes[0].set_xlabel('Recall')
                axes[0].set_ylabel('Precision')
                axes[0].plot(recall_list, precision_list, label=name)
                axes[0].grid(True)
                axes[0].legend()

        if use_plotly:
            margin = 0.01
            fig.layout.yaxis.range = [0 - margin, 1 + margin]
            fig.layout.xaxis.range = [0 - margin, 1 + margin]

            fig.layout.yaxis.title = 'Precision'
            fig.layout.xaxis.title = 'Recall'

            fig.update_layout(height=600, width=1200)
            fig.show()
        else:
            plt.show()

    def draw_ann(self, draw, ann, color, width=5):
        if self.iouType == 'bbox':
            x1, y1, w, h = ann['bbox']
            draw.rectangle([x1, y1, x1+w, y1+h], outline=color, width=width)
        else:
            for poly in ann['segmentation']:
                if len(poly) > 3:
                    draw.line(poly, width=width, fill=color, joint='curve')

    def plot_img(self, img, force_matplot=False, figsize=None, slider=False):
        if plotly_available and not force_matplot and slider:
            fig = px.imshow(img, animation_frame=0,
                            binary_compression_level=5,
                            binary_format='jpg',
                            aspect='auto',
                            labels=dict(animation_frame="shown picture"))

            fig.update_layout(height=700, width=900)
            fig.update_layout(autosize=True)
            fig.show()

        else:
            is_pillow = 'Image' in str(type(img))
            if is_pillow:
                img = [img]
                count = 1
            elif type(img) is list:
                count = len(img)
            else:
                is_batch = len(img.shape) == 4
                if not is_batch:
                    img = np.array([img])
                count = img.shape[0]

            for img_i in range(count):
                if figsize is not None:
                    plt.figure(figsize=figsize)
                plt.imshow(img[img_i], interpolation='nearest')
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

    def display_tp_fp_fn(self, image_ids=['all'],
                         line_width=7,
                         display_fp=True,
                         display_fn=True,
                         display_tp=True,
                         display_gt=True,
                         resize_out_image=None,
                         data_folder=None,
                         categories=None,
                         return_img=False,
                         ):
        image_batch = []

        for image_id, gt_anns in self.cocoGt.imgToAnns.items():
            if (image_id in image_ids) or 'all' in image_ids:
                image = self.cocoGt.imgs[image_id]

                if data_folder is not None:
                    image_fn = osp.join(data_folder, image["file_name"])
                else:
                    image_fn = image["file_name"]

                if osp.exists(image_fn):
                    im = Image.open(image_fn).convert("RGB")
                else:
                    logger.warning(
                        f'[{image_fn}] not found!\nLoading default empty image')

                    im = Image.new("RGB", (image['width'], image['height']))

                mask = Image.new("RGBA", im.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(mask)

                gt_anns = {ann['id']: ann for ann in gt_anns}
                if len(gt_anns) > 0:
                    for ann in gt_anns.values():
                        if categories is None or ann['category_id'] in categories:
                            is_fn = ann.get('fn', False)

                            if is_fn and display_fn:
                                self.draw_ann(
                                    draw, ann, color=self.FN_COLOR, width=line_width)
                            elif display_gt:
                                self.draw_ann(
                                    draw, ann, color=self.GT_COLOR, width=line_width)

                dt_anns = self.cocoDt.imgToAnns[image_id]
                dt_anns = {ann['id']: ann for ann in dt_anns}

                if len(dt_anns) > 0:
                    for ann in dt_anns.values():
                        if categories is None or ann['category_id'] in categories:
                            if ann.get('tp', False):
                                if display_tp:
                                    self.draw_ann(
                                        draw, ann, color=self.DT_COLOR, width=line_width)
                            else:
                                if display_fp:
                                    self.draw_ann(
                                        draw, ann, color=self.FP_COLOR, width=line_width)

                im.paste(mask, mask)
                image_batch.append(im)

        if len(image_batch) >= 1 and resize_out_image is None:
            resize_out_image = image_batch[0].size

        if return_img:
            return image_batch

        if len(image_batch) == 1:
            self.plot_img(np.array(image_batch[0].resize(resize_out_image)))
        elif len(image_batch) > 1:
            image_batch = np.array(
                [np.array(image.resize(resize_out_image)) for image in image_batch])
            self.plot_img(image_batch, slider=True)

    def _compute_confusion_matrix(self, y_true, y_pred, fp={}, fn={}):
        """
        return classes*(classes + fp col + fn col)
        """
        categories_real_ids = list(self.cocoGt.cats)
        categories_enum_ids = {category_id: _i for _i,
                               category_id in enumerate(categories_real_ids)}
        K = len(categories_enum_ids)

        cm = np.zeros((K, K + 2), dtype=np.int32)
        for a, p in zip(y_true, y_pred):
            cm[categories_enum_ids[a]][categories_enum_ids[p]] += 1

        for enum_id, category_id in enumerate(categories_real_ids):
            cm[enum_id][-2] = fp.get(category_id, 0)
            cm[enum_id][-1] = fn.get(category_id, 0)

        return cm

    def compute_confusion_matrix(self):
        if self.useCats:
            logger.warning(
                f"The calculation may not be accurate. No intersection of classes. {self.useCats=}")

        y_true = []
        y_pred = []

        fn = {}
        fp = {}

        for image_id, gt_anns in self.cocoGt.imgToAnns.items():
            gt_anns = {ann['id']: ann for ann in gt_anns}
            if len(gt_anns) > 0:
                for ann in gt_anns.values():
                    if ann.get('fn', False):
                        if fn.get(ann['category_id']) is None:
                            fn[ann['category_id']] = 0

                        fn[ann['category_id']] += 1

            dt_anns = self.cocoDt.imgToAnns[image_id]
            dt_anns = {ann['id']: ann for ann in dt_anns}

            if len(dt_anns) > 0:
                for ann in dt_anns.values():
                    if ann.get('tp', False):
                        y_true.append(gt_anns[ann['gt_id']]['category_id'])
                        y_pred.append(ann['category_id'])
                    else:
                        if fp.get(ann['category_id']) is None:
                            fp[ann['category_id']] = 0

                        fp[ann['category_id']] += 1

        # classes fp fn
        cm = self._compute_confusion_matrix(y_true, y_pred, fp=fp, fn=fn)
        return cm

    def compute_tp_iou(self, categories=None):
        g = []
        d = []
        s = []

        for dt_id, dt_ann in self.cocoDt.anns.items():
            if dt_ann.get('tp', False):
                gt_ann = self.cocoGt.anns[dt_ann['gt_id']]
                if categories is None or gt_ann['category_id'] in categories:
                    s.append(dt_ann.get('score', 1))
                    if self.iouType == 'segm':
                        g.append(gt_ann['rle'])
                        d.append(dt_ann['rle'])
                    elif self.iouType == 'bbox':
                        g.append(gt_ann['bbox'])
                        d.append(dt_ann['bbox'])
                    else:
                        raise Exception('unknown iouType for iou computation')

        iscrowd = [0 for o in g]
        ious = maskUtils.iou(d, g, iscrowd).diagonal()
        return ious

    def display_matrix(self, in_percent=False, conf_matrix=None, figsize=(10, 10), fontsize=16):
        if conf_matrix is None:
            conf_matrix = self.compute_confusion_matrix()

        names = [category['name']
                 for category_id, category in self.cocoGt.cats.items()]
        names += ['fp', 'fn']

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
        plt.yticks(list(range(len(names[:-2]))), names[:-2])

        title = 'Confusion Matrix'
        if in_percent:
            title += ' [%]'

        plt.title(title, fontsize=fontsize)
        plt.show()
