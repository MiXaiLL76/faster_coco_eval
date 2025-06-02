import logging
from typing import Dict, List, Optional, Union

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
        gt_ann_ids: Optional[set] = None,
        dt_ann_ids: Optional[set] = None,
        return_fig: bool = False,
    ):
        """Display the image with the results.

        Args:
            image_id (int): Image id.
            display_fp (bool): Display false positives.
            display_fn (bool): Display false negatives.
            display_tp (bool): Display true positives.
            display_gt (bool): Display ground truth.
            data_folder (Optional[str]): Data folder.
            categories (Optional[list]): Categories to display.
            gt_ann_ids (Optional[set]): Ground truth annotation ids.
            dt_ann_ids (Optional[set]): Detected annotation ids.
            return_fig (bool): Return the figure.

        Returns:
            Optional[Any]: The figure object if return_fig is True, otherwise None.
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
            gt_ann_ids=gt_ann_ids,
            dt_ann_ids=dt_ann_ids,
            return_fig=return_fig,
        )

    def display_tp_fp_fn(
        self,
        image_ids: Union[List[int], List[str]] = ["all"],
        display_fp: bool = True,
        display_fn: bool = True,
        display_tp: bool = True,
        display_gt: bool = False,
        data_folder: Optional[str] = None,
        categories: Optional[list] = None,
    ):
        """Display true positives, false positives, and false negatives for
        given images.

        Args:
            image_ids (Union[List[int], List[str]]): List of image ids or ["all"] to display all images.
            display_fp (bool): Display false positives.
            display_fn (bool): Display false negatives.
            display_tp (bool): Display true positives.
            display_gt (bool): Display ground truth.
            data_folder (Optional[str]): Data folder.
            categories (Optional[list]): Categories to display.

        Returns:
            None
        """
        if image_ids == ["all"]:
            image_ids = list(self.cocoGt.imgs.keys())

        for image_id in image_ids:
            if self.cocoGt.imgs.get(image_id) is None:
                logger.warning(f"Image {image_id} not found")
                continue

            self.display_image(
                image_id,
                display_fp=display_fp,
                display_fn=display_fn,
                display_tp=display_tp,
                display_gt=display_gt,
                data_folder=data_folder,
                categories=categories,
            )

    def _compute_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        fp: Dict[int, int] = {},
        fn: Dict[int, int] = {},
    ) -> np.ndarray:
        """Compute the confusion matrix.

        Args:
            y_true (List[int]): True labels (category ids).
            y_pred (List[int]): Predicted labels (category ids).
            fp (Dict[int, int]): False positive counts per category id.
            fn (Dict[int, int]): False negative counts per category id.

        Returns:
            np.ndarray: The confusion matrix.
        """
        categories_real_ids = list(self.cocoGt.cats)
        categories_enum_ids = {category_id: _i for _i, category_id in enumerate(categories_real_ids)}
        K = len(categories_enum_ids)

        cm = np.zeros((K, K + 2), dtype=np.float32)
        for a, p in zip(y_true, y_pred):
            cm[categories_enum_ids[a]][categories_enum_ids[p]] += 1

        for enum_id, category_id in enumerate(categories_real_ids):
            cm[enum_id][-2] = fp.get(category_id, 0)
            cm[enum_id][-1] = fn.get(category_id, 0)

        return cm

    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute the confusion matrix for the current evaluation.

        Returns:
            np.ndarray: The confusion matrix.

        Raises:
            AssertionError: If self.eval is None (evaluate() was not run).
        """
        assert self.eval is not None, "Run first self.evaluate()"

        if self.useCats:
            logger.warning(f"The calculation may not be accurate. No intersection of classes. useCats={self.useCats}")

        y_true = []
        y_pred = []

        fn = {}
        fp = {}

        for _, ann in self.cocoGt.anns.items():
            if ann.get("dt_id") is not None:
                dt_ann = self.cocoDt.anns[ann["dt_id"]]

                y_true.append(ann["category_id"])
                y_pred.append(dt_ann["category_id"])

            else:
                if fn.get(ann["category_id"]) is None:
                    fn[ann["category_id"]] = 0
                fn[ann["category_id"]] += 1

        for _, ann in self.cocoDt.anns.items():
            if ann.get("gt_id") is None:
                if fp.get(ann["category_id"]) is None:
                    fp[ann["category_id"]] = 0
                fp[ann["category_id"]] += 1

        # classes fp fn
        cm = self._compute_confusion_matrix(y_true, y_pred, fp=fp, fn=fn)
        return cm

    def display_matrix(
        self,
        normalize: bool = False,
        conf_matrix: Optional[np.ndarray] = None,
        return_fig: bool = False,
    ):
        """Display the confusion matrix.

        Args:
            normalize (bool): Normalize the matrix.
            conf_matrix (Optional[np.ndarray]): Confusion matrix to display. If None, compute it.
            return_fig (bool): Return the figure.

        Returns:
            Optional[Any]: The figure object if return_fig is True, otherwise None.
        """
        if conf_matrix is None:
            conf_matrix = self.compute_confusion_matrix()

        labels = [category["name"] for _, category in self.cocoGt.cats.items()]

        return display_matrix(conf_matrix, labels, normalize=normalize, return_fig=return_fig)
