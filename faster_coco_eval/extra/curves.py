import logging

import numpy as np

from ..core import COCOeval_faster
from .draw import plot_ced_metric, plot_f1_confidence, plot_pre_rec
from .extra import ExtraEval

logger = logging.getLogger(__name__)


class Curves(ExtraEval):
    def build_curve(self, label: str):
        """Build the curve for a given label."""
        assert self.eval is not None, "Run first self.evaluate()"

        curve = []

        if self.useCats:
            cat_ids = list(range(self.eval["precision"].shape[2]))
        else:
            cat_ids = [0]

        for category_id in cat_ids:
            _label = "[{}={}] ".format(label, category_id)
            if len(cat_ids) == 1:
                _label = ""

            precision_list = self.eval["precision"][
                :, :, category_id, :, :
            ].ravel()
            recall_list = self.recThrs
            scores = self.eval["scores"][:, :, category_id, :, :].ravel()
            auc = round(
                COCOeval_faster.calc_auc(recall_list, precision_list), 4
            )

            curve.append(
                dict(
                    recall_list=recall_list,
                    precision_list=precision_list,
                    name="{}auc: {:.3f}".format(_label, auc),
                    label=_label,
                    scores=scores,
                    auc=auc,
                    category_id=category_id,
                )
            )

        return curve

    def plot_pre_rec(
        self, curves=None, label: str = "category_id", return_fig: bool = False
    ):
        """Plot the precision-recall curve.

        curves: list of curves to plot
        label: label of the curves
        return_fig: return the figure

        """
        if curves is None:
            curves = self.build_curve(label)

        return plot_pre_rec(curves, return_fig=return_fig)

    def plot_f1_confidence(
        self, curves=None, label: str = "category_id", return_fig: bool = False
    ):
        """Plot the F1 confidence curve.

        curves: list of curves to plot
        label: label of the curves
        return_fig: return the figure

        """
        if curves is None:
            curves = self.build_curve(label)

        return plot_f1_confidence(curves, return_fig=return_fig)

    def build_ced_curve(self, mse_count: int = 1000):
        """Build the curve for all categories."""
        assert self.eval is not None, "Run first self.evaluate()"

        curves = []
        for category_id, category in self.cocoGt.cats.items():
            all_mse = []
            for ann_id in self.cocoGt.get_ann_ids(cat_ids=[category_id]):
                gt_ann = self.cocoGt.anns[ann_id]
                if gt_ann.get("keypoints", False) and gt_ann.get(
                    "matched", False
                ):
                    dt_ann = self.cocoDt.anns[gt_ann["dt_id"]]

                    # https://en.wikipedia.org/wiki/Mean_squared_error
                    if self.iouType == "keypoints":
                        gt_kps = np.array(gt_ann["keypoints"]).reshape(-1, 3)[
                            :, :2
                        ]
                        dt_kps = np.array(dt_ann["keypoints"]).reshape(-1, 3)[
                            :, :2
                        ]
                    elif self.iouType == "bbox":
                        gt_kps = np.array(gt_ann["bbox"])  # xywh
                        gt_kps[2:] += gt_kps[:2]  # xyxy
                        dt_kps = np.array(dt_ann["bbox"])  # xywh
                        dt_kps[2:] += dt_kps[:2]  # xyxy
                    else:
                        raise ValueError(
                            f"not supported iouType {self.iouType} for CED"
                        )

                    mse = np.square(gt_kps - dt_kps).mean()
                    all_mse.append(mse)

            if len(all_mse) == 0:
                continue

            all_mse = np.array(all_mse)
            _median = np.median(all_mse)
            _q3 = np.sqrt(np.var(all_mse))

            _curve = {
                "mse": [0],
                "count": [0],
                "total_count": len(all_mse),
                "category": category,
            }

            for max_mse in np.linspace(
                all_mse.min(), (_median + _q3), mse_count
            ):
                _mask = all_mse < max_mse
                _curve["count"].append(_mask.sum())
                _curve["mse"].append(max_mse)

            _curve["count"].append(_curve["total_count"])
            _curve["mse"].append(all_mse.max())
            curves.append(_curve)
        return curves

    def plot_ced_metric(
        self, curves=None, normalize: bool = True, return_fig: bool = False
    ):
        """Plot the CED metric curve.

        curves: list of curves to plot
        normalize: normalize the curve
        return_fig: return the figure

        """
        if curves is None:
            curves = self.build_ced_curve()

        return plot_ced_metric(
            curves, normalize=normalize, return_fig=return_fig
        )
