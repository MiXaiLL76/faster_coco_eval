import logging
import plotly.graph_objects as go

from ..core import COCOeval_faster
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

        fig = go.Figure()

        for _curve in curves:
            recall_list = _curve["recall_list"]
            precision_list = _curve["precision_list"]
            scores = _curve["scores"]
            name = _curve["name"]

            fig.add_trace(
                go.Scatter(
                    x=recall_list,
                    y=precision_list,
                    name=name,
                    text=scores,
                    hovertemplate="Pre: %{y:.3f}<br>"
                    + "Rec: %{x:.3f}<br>"
                    + "Score: %{text:.3f}<extra></extra>",
                    showlegend=True,
                    mode="lines",
                )
            )

        margin = 0.01
        fig.layout.yaxis.range = [0 - margin, 1 + margin]
        fig.layout.xaxis.range = [0 - margin, 1 + margin]

        fig.layout.yaxis.title = "Precision"
        fig.layout.xaxis.title = "Recall"

        fig.update_xaxes(showspikes=True)
        fig.update_yaxes(showspikes=True)

        layout = {
            "title": "Precision-Recall",
            "autosize": True,
            "height": 600,
            "width": 1200,
        }

        fig.update_layout(layout)

        if return_fig:
            return fig

        fig.show()
