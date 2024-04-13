import logging

from ..core import COCOeval_faster
from .draw import plot_f1_confidence, plot_pre_rec
from .extra import ExtraEval

logger = logging.getLogger(__name__)


class Curves(ExtraEval):
    def build_curve(self, label):
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
        if curves is None:
            curves = self.build_curve(label)

        return plot_pre_rec(curves, return_fig=return_fig)

    def plot_f1_confidence(
        self, curves=None, label: str = "category_id", return_fig: bool = False
    ):
        if curves is None:
            curves = self.build_curve(label)

        return plot_f1_confidence(curves, return_fig=return_fig)
