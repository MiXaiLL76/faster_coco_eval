import logging
import os.path as osp
from typing import Optional

import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from PIL import Image

from ..core import COCO

logger = logging.getLogger(__name__)

A = 0.1
DT_COLOR = (238, 130, 238, A)

GT_COLOR = (0, 255, 0, A)
FN_COLOR = (0, 0, 255, A)
FP_COLOR = (255, 0, 0, A)


def generate_ann_polygon(
    ann: dict,
    color: tuple,
    iouType: str = "bbox",
    text: Optional[str] = None,
    legendgroup: Optional[str] = None,
):
    all_x = []
    all_y = []

    if iouType == "bbox":
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
    cocoGt: COCO,
    cocoDt: Optional[COCO] = None,
    image_id: int = 1,
    iouType: Optional[str] = "bbox",
    display_fp: bool = True,
    display_fn: bool = True,
    display_tp: bool = True,
    display_gt: bool = True,
    data_folder: Optional[str] = None,
    categories: Optional[list] = None,
    return_fig: bool = False,
):
    polygons = []

    image = cocoGt.imgs[image_id]
    gt_anns = {ann["id"]: ann for ann in cocoGt.imgToAnns[image_id]}

    if cocoDt is not None:
        dt_anns = {ann["id"]: ann for ann in cocoDt.imgToAnns[image_id]}
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
        category["id"]: category["name"] for _, category in cocoGt.cats.items()
    }

    if len(gt_anns) > 0:
        for ann in gt_anns.values():
            if categories is None or ann["category_id"] in categories:
                if ann.get("fn", False):
                    if display_fn:
                        poly = generate_ann_polygon(
                            ann,
                            color=FN_COLOR,
                            iouType=iouType,
                            text="<b>FN</b><br>id={}<br>category={}".format(
                                ann["id"],
                                categories_labels[ann["category_id"]],
                            ),
                            legendgroup="fn",
                        )
                        polygons.append(poly)
                else:
                    if display_gt:
                        poly = generate_ann_polygon(
                            ann,
                            color=GT_COLOR,
                            iouType=iouType,
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
                        poly = generate_ann_polygon(
                            ann,
                            color=DT_COLOR,
                            iouType=iouType,
                            text=(
                                "<b>DT</b><br>"
                                "id={}<br>"
                                "category={}<br>"
                                "score={:.2f}<br>"
                                "IoU={:.2f}"
                            ).format(
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
                        poly = generate_ann_polygon(
                            ann,
                            color=FP_COLOR,
                            iouType=iouType,
                            text=(
                                "<b>FP</b><br>"
                                "id={}<br>"
                                "category={}<br>"
                                "score={:.2f}"
                            ).format(
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


def display_matrix(
    conf_matrix: np.ndarray,
    labels: list,
    normalize: bool = False,
    return_fig: bool = False,
):
    _labels = labels + ["fp", "fn"]

    if normalize:
        conf_matrix /= conf_matrix.sum(axis=1).reshape(-1, 1)
        conf_matrix *= 100

    hovertemplate = "Real: %{y}<br>Predict: %{x}<br>"

    if normalize:
        hovertemplate += "Percent: %{z:.0f}<extra></extra>"
    else:
        hovertemplate += "Count: %{z:.0f}<extra></extra>"

    heatmap = go.Heatmap(
        z=conf_matrix,
        x=_labels,
        y=_labels[:-2],
        colorscale="Blues",
        hovertemplate=hovertemplate,
    )

    annotations = []
    for j, row in enumerate(conf_matrix):
        for i, value in enumerate(row):
            text_value = "{:.0f}".format(value)
            if normalize:
                text_value += "%"

            annotations.append(
                {
                    "x": _labels[i],
                    "y": _labels[j],
                    "font": {"color": "white"},
                    "text": text_value,
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False,
                }
            )

    title = "Confusion Matrix"
    if normalize:
        title = "Normalized " + title

    layout = {
        "title": title,
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


def plot_pre_rec(curves, return_fig: bool = False):
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


def plot_f1_confidence(curves, return_fig: bool = False):
    fig = go.Figure()
    eps = 1e-16
    for _curve in curves:
        recall_list = _curve["recall_list"]
        precision_list = _curve["precision_list"]
        scores = _curve["scores"]
        f1_curve = (
            2
            * precision_list
            * recall_list
            / (precision_list + recall_list + eps)
        )

        name = _curve["label"] if len(_curve["label"]) > 0 else "F1-Confidence"

        fig.add_trace(
            go.Scatter(
                x=scores,
                y=f1_curve,
                name=name,
                hovertemplate="F1: %{y:.3f}<br>"
                + "Confidence: %{x:.3f}<br><extra></extra>",
                showlegend=True,
                mode="lines",
            )
        )

    margin = 0.01
    fig.layout.yaxis.range = [0 - margin, 1 + margin]
    fig.layout.xaxis.range = [0 - margin, 1 + margin]

    fig.layout.yaxis.title = "F1"
    fig.layout.xaxis.title = "Confidence"

    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)

    layout = {
        "title": "F1-Confidence",
        "autosize": True,
        "height": 600,
        "width": 1200,
    }

    fig.update_layout(layout)

    if return_fig:
        return fig

    fig.show()
