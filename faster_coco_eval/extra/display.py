import logging
from typing import Optional

import numpy as np

from .draw import display_image, display_matrix
from .extra import ExtraEval

logger = logging.getLogger(__name__)


class PreviewResults(ExtraEval):
    def display_image(
        self,
        image_id: int = 1,
        display_fp: bool = True,
        display_fn: bool = True,
        display_tp: bool = True,
        display_gt: bool = True,
        data_folder: Optional[str] = None,
        categories: Optional[list] = None,
        return_fig: bool = False,
    ):
        """
        Display the image with the results
        image_id: image id
        display_fp: display false positive
        display_fn: display false negative
        display_tp: display true positive
        display_gt: display ground truth
        data_folder: data folder
        categories: categories to display
        return_fig: return the figure
        """
        return display_image(
            self.cocoGt,
            self.cocoDt,
            image_id=image_id,
            iouType=self.iouType,
            display_fp=display_fp,
            display_fn=display_fn,
            display_tp=display_tp,
            display_gt=display_gt,
            data_folder=data_folder,
            categories=categories,
            return_fig=return_fig,
        )

    def display_tp_fp_fn(
        self,
        image_ids=["all"],
        display_fp=True,
        display_fn=True,
        display_tp=True,
        display_gt=False,
        data_folder=None,
        categories=None,
    ):
        for image_id, _ in self.cocoGt.imgToAnns.items():
            if (image_id in image_ids) or "all" in image_ids:
                self.display_image(
                    image_id,
                    display_fp=display_fp,
                    display_fn=display_fn,
                    display_tp=display_tp,
                    display_gt=display_gt,
                    data_folder=data_folder,
                    categories=categories,
                )

    def _compute_confusion_matrix(self, y_true, y_pred, fp={}, fn={}):
        """
        Compute the confusion matrix
        y_true: true labels
        y_pred: predicted labels
        fp: false positive
        fn: false negative
        """
        categories_real_ids = list(self.cocoGt.cats)
        categories_enum_ids = {
            category_id: _i
            for _i, category_id in enumerate(categories_real_ids)
        }
        K = len(categories_enum_ids)

        cm = np.zeros((K, K + 2), dtype=np.float32)
        for a, p in zip(y_true, y_pred):
            cm[categories_enum_ids[a]][categories_enum_ids[p]] += 1

        for enum_id, category_id in enumerate(categories_real_ids):
            cm[enum_id][-2] = fp.get(category_id, 0)
            cm[enum_id][-1] = fn.get(category_id, 0)

        return cm

    def compute_confusion_matrix(self):
        """Compute the confusion matrix."""
        assert self.eval is not None, "Run first self.evaluate()"

        if self.useCats:
            logger.warning(
                "The calculation may not be accurate. No intersection of"
                " classes. useCats={}".format(self.useCats)
            )

        y_true = []
        y_pred = []

        fn = {}
        fp = {}

        for ann_id, ann in self.cocoGt.anns.items():
            if ann.get("dt_id") is not None:
                dt_ann = self.cocoDt.anns[ann["dt_id"]]

                y_true.append(ann["category_id"])
                y_pred.append(dt_ann["category_id"])

            else:
                if fn.get(ann["category_id"]) is None:
                    fn[ann["category_id"]] = 0
                fn[ann["category_id"]] += 1

        for ann_id, ann in self.cocoDt.anns.items():
            if ann.get("gt_id") is None:
                if fp.get(ann["category_id"]) is None:
                    fp[ann["category_id"]] = 0
                fp[ann["category_id"]] += 1

        # classes fp fn
        cm = self._compute_confusion_matrix(y_true, y_pred, fp=fp, fn=fn)
        return cm

    def display_matrix(
        self, normalize=False, conf_matrix=None, return_fig: bool = False
    ):
        """
        Display the confusion matrix
        normalize: normalize the matrix
        conf_matrix: confusion matrix to display
        return_fig: return the figure
        """
        if conf_matrix is None:
            conf_matrix = self.compute_confusion_matrix()

        labels = [category["name"] for _, category in self.cocoGt.cats.items()]

        return display_matrix(
            conf_matrix, labels, normalize=normalize, return_fig=return_fig
        )
