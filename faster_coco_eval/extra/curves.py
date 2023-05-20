from ..core.faster_eval_api import COCOeval_faster
from .extra import ExtraEval

import numpy as np
import logging

import matplotlib.pyplot as plt

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    plotly_available = True
except:
    plotly_available = False

logger = logging.getLogger(__name__)

class Curves(ExtraEval):
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
            auc = round(COCOeval_faster.calc_auc(recall_list, precision_list), 4)

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