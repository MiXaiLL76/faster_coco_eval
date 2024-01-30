import logging
import numpy as np
import os.path as osp
import plotly.express as px
import plotly.graph_objs as go
from PIL import Image

from .extra import ExtraEval

logger = logging.getLogger(__name__)


class PreviewResults(ExtraEval):
    A = 0.1
    DT_COLOR = (238, 130, 238, A)

    GT_COLOR = (0, 255, 0, A)
    FN_COLOR = (0, 0, 255, A)
    FP_COLOR = (255, 0, 0, A)

    def get_ann_poly(self, ann, color, text=None, legendgroup=None):
        all_x = []
        all_y = []

        if self.iouType == "bbox":
            x1, y1, w, h = ann["bbox"]
            all_x = [x1, x1 + w, x1 + w, x1, x1, None]
            all_y = [y1, y1, y1 + h, y1 + h, y1, None]
        else:
            for poly in ann["segmentation"]:
                if len(poly) > 3:
                    poly += poly[:2]
                    poly = np.array(poly).reshape(-1, 2)
                    all_x += poly[:, 0].tolist() + [None]
                    all_y += poly[:, 1].tolist() + [None]

        return go.Scatter(
            x=all_x,
            y=all_y,
            name="",
            text=text,
            hovertemplate="{text}<extra></extra>",
            mode="lines",
            legendgroup=legendgroup,
            legendgrouptitle_text=legendgroup,
            showlegend=False,
            fill="toself",
            fillcolor="rgba{}".format(color),
            line=dict(color="rgb{}".format(color[:3])),
        )

    def display_image(
        self,
        image_id=1,
        display_fp=True,
        display_fn=True,
        display_tp=True,
        display_gt=True,
        data_folder=None,
        categories=None,
        return_fig: bool = False,
    ):
        polygons = []

        image = self.cocoGt.imgs[image_id]
        gt_anns = {ann["id"]: ann for ann in self.cocoGt.imgToAnns[image_id]}

        if self.cocoDt is not None:
            dt_anns = {
                ann["id"]: ann for ann in self.cocoDt.imgToAnns[image_id]
            }
        else:
            dt_anns = {}

        image_fn = image["file_name"]
        if data_folder is not None:
            image_load_path = osp.join(data_folder, image["file_name"])
        else:
            image_load_path = image["file_name"]

        if osp.exists(image_load_path):
            im = Image.open(image_load_path).convert("RGB")
        else:
            logger.warning(
                "[{}] not found!\nLoading default empty image".format(
                    image_load_path
                )
            )

            im = Image.new("RGB", (image["width"], image["height"]))

        categories_labels = {
            category["id"]: category["name"]
            for _, category in self.cocoGt.cats.items()
        }

        if len(gt_anns) > 0:
            for ann in gt_anns.values():
                if categories is None or ann["category_id"] in categories:
                    if ann.get("fn", False):
                        if display_fn:
                            poly = self.get_ann_poly(
                                ann,
                                color=self.FN_COLOR,
                                text="<b>FN</b><br>id={}<br>category={}".format(
                                    ann["id"],
                                    categories_labels[ann["category_id"]],
                                ),
                                legendgroup="fn",
                            )
                            polygons.append(poly)
                    else:
                        if display_gt:
                            poly = self.get_ann_poly(
                                ann,
                                color=self.GT_COLOR,
                                text="<b>GT</b><br>id={}<br>category={}".format(
                                    ann["id"],
                                    categories_labels[ann["category_id"]],
                                ),
                                legendgroup="gt",
                            )
                            polygons.append(poly)

        if len(dt_anns) > 0:
            for ann in dt_anns.values():
                if categories is None or ann["category_id"] in categories:
                    if ann.get("tp", False):
                        if display_tp:
                            poly = self.get_ann_poly(
                                ann,
                                color=self.DT_COLOR,
                                text="<b>DT</b><br>id={}<br>category={}<br>score={:.2f}<br>IoU={:.2f}".format(
                                    ann["id"],
                                    categories_labels[ann["category_id"]],
                                    ann["score"],
                                    ann["iou"],
                                ),
                                legendgroup="tp",
                            )
                            polygons.append(poly)
                    else:
                        if display_fp:
                            poly = self.get_ann_poly(
                                ann,
                                color=self.FP_COLOR,
                                text="<b>FP</b><br>id={}<br>category={}<br>score={:.2f}".format(
                                    ann["id"],
                                    categories_labels[ann["category_id"]],
                                    ann["score"],
                                ),
                                legendgroup="fp",
                            )
                            polygons.append(poly)

        fig = px.imshow(
            im,
            binary_compression_level=5,
            binary_format="jpg",
            aspect="auto",
            labels=dict(animation_frame="shown picture"),
        )

        legends = {}
        for poly in polygons:
            if legends.get(poly.legendgroup) is None:
                poly.showlegend = True
                legends[poly.legendgroup] = True

            fig.add_trace(poly)

        layout = {
            "title": "image_id={}<br>image_fn={}".format(image_id, image_fn),
            "autosize": True,
            "height": 700,
            "width": 900,
        }

        fig.update_layout(layout)
        fig.update_xaxes(range=[0, image["width"]])
        fig.update_yaxes(range=[image["height"], 0])

        if return_fig:
            return fig

        fig.show()

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
        return classes*(classes + fp col + fn col)
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
        assert self.eval is not None, "Run first self.evaluate()"

        if self.useCats:
            logger.warning(
                "The calculation may not be accurate. No intersection of classes. useCats={}".format(
                    self.useCats
                )
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
        self, in_percent=False, conf_matrix=None, return_fig: bool = False
    ):
        if conf_matrix is None:
            conf_matrix = self.compute_confusion_matrix()

        labels = [category["name"] for _, category in self.cocoGt.cats.items()]
        labels += ["fp", "fn"]

        if in_percent:
            conf_matrix /= conf_matrix.sum(axis=1).reshape(-1, 1)
            conf_matrix *= 100

        hovertemplate = "Real: %{y}<br>" "Predict: %{x}<br>"

        if in_percent:
            hovertemplate += "Percent: %{z:.0f}<extra></extra>"
        else:
            hovertemplate += "Count: %{z:.0f}<extra></extra>"

        heatmap = go.Heatmap(
            z=conf_matrix,
            x=labels,
            y=labels[:-2],
            colorscale="Blues",
            hovertemplate=hovertemplate,
        )

        annotations = []
        for j, row in enumerate(conf_matrix):
            for i, value in enumerate(row):
                text_value = "{:.0f}".format(value)
                if in_percent:
                    text_value += "%"

                annotations.append(
                    {
                        "x": labels[i],
                        "y": labels[j],
                        "font": {"color": "white"},
                        "text": text_value,
                        "xref": "x1",
                        "yref": "y1",
                        "showarrow": False,
                    }
                )

        layout = {
            "title": "Confusion Matrix",
            "xaxis": {"title": "Predicted value"},
            "yaxis": {"title": "Real value"},
            "annotations": annotations,
        }

        fig = go.Figure(data=[heatmap], layout=layout)
        fig.update_traces(showscale=False)
        fig.update_layout(height=700, width=900)

        if return_fig:
            return fig

        fig.show()
