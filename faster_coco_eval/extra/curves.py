import logging

import numpy as np

from ..core import COCOeval_faster
from .draw import plot_ced_metric, plot_f1_confidence, plot_pre_rec
from .extra import ExtraEval

logger = logging.getLogger(__name__)


class Curves(ExtraEval):
    def build_curve(self, label: str) -> list[dict]:
        """Build the curve for a given label.

        Args:
            label (str): The label to build the curve for.

        Returns:
            List[dict]: A list of dictionaries containing
                the curve data for each category.

        Raises:
            AssertionError: If self.eval is None (evaluate() was not called).
        """
        assert self.eval is not None, "Run first self.evaluate()"

        curve = []

        if self.useCats:
            cat_ids = list(range(self.eval["precision"].shape[2]))
            if hasattr(self, "params") and self.params is not None and self.params.catIds:
                real_category_ids = list(self.params.catIds)
            else:
                real_category_ids = sorted(self.cocoGt.cats.keys())
        else:
            cat_ids = [0]
            real_category_ids = []

        for category_index in cat_ids:
            category_id = real_category_ids[category_index] if self.useCats else category_index
            _label = f"[{label}={category_id}] "
            if len(cat_ids) == 1:
                _label = ""

            precision_list = self.eval["precision"][:, :, category_index, :, :].ravel()
            recall_list = np.asarray(self.recThrs).ravel()
            scores = self.eval["scores"][:, :, category_index, :, :].ravel()
            point_count = min(len(recall_list), len(precision_list), len(scores))
            recall_list = recall_list[:point_count]
            precision_list = precision_list[:point_count]
            scores = scores[:point_count]
            valid = precision_list > -1
            if not np.any(valid):
                logger.warning("Skipping category %s: precision contains no valid values", category_id)
                continue

            recall_list = recall_list[valid]
            precision_list = precision_list[valid]
            scores = scores[valid]
            auc = round(COCOeval_faster.calc_auc(recall_list, precision_list), 4)

            curve.append(
                dict(
                    recall_list=recall_list,
                    precision_list=precision_list,
                    name=f"{_label}auc: {auc:.3f}",
                    label=_label,
                    scores=scores,
                    auc=auc,
                    category_id=category_id,
                )
            )

        return curve

    def plot_pre_rec(
        self,
        curves: list[dict] | None = None,
        label: str | None = "category_id",
        return_fig: bool | None = False,
    ):
        """Plot the precision-recall curve.

        Args:
            curves (Optional[List[dict]], optional): List of curves to plot.
                If None, it will build the curves. Defaults to None.
            label (Optional[str], optional): Label for the curves. Defaults to "category_id".
            return_fig (Optional[bool], optional): Return the figure object. Defaults to False.

        Returns:
            plotly.graph_objs._figure.Figure or None: The figure object if return_fig is True, otherwise None.
        """
        if curves is None:
            curves = self.build_curve(label)

        return plot_pre_rec(curves, return_fig=return_fig)

    def plot_f1_confidence(
        self,
        curves: list[dict] | None = None,
        label: str | None = "category_id",
        return_fig: bool | None = False,
    ):
        """Plot the F1 confidence curve.

        Args:
            curves (Optional[List[dict]], optional): List of curves to plot.
                If None, it will build the curves. Defaults to None.
            label (Optional[str], optional): Label for the curves. Defaults to "category_id".
            return_fig (Optional[bool], optional): Return the figure object. Defaults to False.

        Returns:
            plotly.graph_objs._figure.Figure or None: The figure object if return_fig is True, otherwise None.
        """
        if curves is None:
            curves = self.build_curve(label)

        return plot_f1_confidence(curves, return_fig=return_fig)

    def build_ced_curve(self, mae_count: int = 1000) -> list[dict]:
        """Build the CED (Cumulative Error Distribution) curve for all
        categories.

        Args:
            mae_count (int, optional): Number of points to use for the CED curve. Defaults to 1000.

        Returns:
            List[dict]: List of dictionaries containing CED curve data for each category.

        Raises:
            AssertionError: If self.eval is None (evaluate() was not called).
            ValueError: If the iouType is not 'keypoints' (other types are not supported).
        """
        assert self.eval is not None, "Run first self.evaluate()"

        curves = []
        for category_id, category in self.cocoGt.cats.items():
            _curve = {
                "all_mae": [],
                "mae": {},
                "category": category,
            }
            for ann_id in self.cocoGt.get_ann_ids(cat_ids=[category_id]):
                gt_ann = self.cocoGt.anns[ann_id]
                if gt_ann.get("keypoints", False) and gt_ann.get("matched", False):
                    dt_ann = self.cocoDt.anns[gt_ann["dt_id"]]

                    if self.iouType == "keypoints":
                        gt_xyv = np.array(gt_ann["keypoints"]).reshape(-1, 3)
                        dt_xyv = np.array(dt_ann["keypoints"]).reshape(-1, 3)

                        dt_ann["mae_keypoints"] = []
                        for _id, kp_name in enumerate(category["keypoints"]):
                            dt_ann["mae_keypoints"].append(
                                np.mean(np.abs(np.subtract(gt_xyv[_id, :2], dt_xyv[_id, :2])))
                            )

                            if _curve["mae"].get(kp_name) is None:
                                _curve["mae"][kp_name] = {
                                    "all_mae": [],
                                }

                            _curve["mae"][kp_name]["all_mae"].append(dt_ann["mae_keypoints"][_id])

                        dt_ann["mae"] = np.mean(dt_ann["mae_keypoints"])
                        _curve["all_mae"].append(dt_ann["mae"])

                    else:
                        raise ValueError(f"not supported iouType {self.iouType} for CED")

            if len(_curve["all_mae"]) == 0:
                continue

            def create_curve(x: list, count: int) -> dict:
                """Create the CED curve data.

                Args:
                    x (list): List of MAE values.
                    count (int): Number of points for the curve.

                Returns:
                    dict: Dictionary containing 'x' and 'y' list for the curve.
                """
                x = np.array(x)
                _median = np.median(x)
                _q3 = np.percentile(x, 75)
                result = {
                    "x": [0],
                    "y": [0],
                }

                curve_limit = min(_median + _q3, x.max())
                if x.min() < curve_limit:
                    for val in np.linspace(x.min(), curve_limit, count):
                        _mask = x < val
                        result["y"].append(_mask.sum())
                        result["x"].append(val)

                result["y"].append(len(x))
                result["x"].append(x.max())
                return result

            all_result = create_curve(_curve["all_mae"], mae_count)
            _curve["mae"]["MEAN"] = {
                "x": all_result["x"],
                "y": all_result["y"],
            }

            for _id, kp_name in enumerate(category["keypoints"]):
                _result = create_curve(_curve["mae"][kp_name]["all_mae"], mae_count)
                _curve["mae"][kp_name]["x"] = _result["x"]
                _curve["mae"][kp_name]["y"] = _result["y"]

            curves.append(_curve)
        return curves

    def plot_ced_metric(
        self,
        curves: list[dict] | None = None,
        normalize: bool | None = True,
        return_fig: bool | None = False,
    ):
        """Plot the CED metric curve.

        Args:
            curves (Optional[List[dict]], optional): List of curves to plot.
                If None, will build the curves. Defaults to None.
            normalize (Optional[bool], optional): Whether to normalize the curve. Defaults to True.
            return_fig (Optional[bool], optional): Return the figure object. Defaults to False.

        Returns:
            plotly.graph_objs._figure.Figure or None: The figure object if return_fig is True, otherwise None.
        """
        if curves is None:
            curves = self.build_ced_curve()

        return plot_ced_metric(curves, normalize=normalize, return_fig=return_fig)
