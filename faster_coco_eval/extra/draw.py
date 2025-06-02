import logging
import os.path as osp
import sys
from typing import List, Optional

if sys.version_info >= (3, 8):
    from typing import Literal

    showAnnsiouTypeT = Literal["segm", "bbox"]
else:
    showAnnsiouTypeT = str

import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from PIL import Image

from ..core import COCO
from .utils import convert_ann_rle_to_poly

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
    category_id_to_skeleton: Optional[dict] = None,
) -> go.Scatter:
    """Generate annotation polygon for plotly.

    Args:
        ann (dict): Annotation dictionary.
        color (tuple): Color of the annotation, as (R, G, B, A) tuple.
        iouType (str, optional): Type of the annotation. One of 'bbox', 'segm', or 'keypoints'. Default is "bbox".
        text (str, optional): Text to display on hover. Default is None.
        legendgroup (str, optional): Legend group to display. Default is None.
        category_id_to_skeleton (dict, optional): Dictionary mapping category_id to skeleton (for keypoints). Default is None.

    Returns:
        go.Scatter: Plotly Scatter object representing the annotation polygon.
    """  # noqa: E501
    all_x = []
    all_y = []

    if iouType == "bbox":
        x1, y1, w, h = ann["bbox"]
        all_x = [x1, x1 + w, x1 + w, x1, x1, None]
        all_y = [y1, y1, y1 + h, y1 + h, y1, None]
    elif iouType == "segm":
        ann = convert_ann_rle_to_poly(ann)

        for poly in ann["segmentation"]:
            if len(poly) > 3:
                poly += poly[:2]
                poly = np.array(poly).reshape(-1, 2)
                all_x += poly[:, 0].tolist() + [None]
                all_y += poly[:, 1].tolist() + [None]
    elif iouType == "keypoints":
        skeleton = category_id_to_skeleton.get(ann.get("category_id"))
        keypoints = ann.get("keypoints")

        if (skeleton is None) or (keypoints is None):
            return

        xyz = np.array(keypoints).reshape(-1, 3)
        ready_bones = {i: True for i in range(xyz.shape[0])}
        for p1, p2 in skeleton:
            if ready_bones.get(p1 - 1, False) and ready_bones.get(p2 - 1, False):
                all_x += [xyz[int(p1 - 1), 0], xyz[int(p2 - 1), 0], None]
                all_y += [xyz[int(p1 - 1), 1], xyz[int(p2 - 1), 1], None]

        if ann.get("bbox"):
            x1, y1, w, h = ann.get("bbox")
            all_x += [x1, x1 + w, x1 + w, x1, x1, None]
            all_y += [y1, y1, y1 + h, y1 + h, y1, None]

    else:
        raise ValueError()

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
        fillcolor=f"rgba{color}",
        line=dict(color=f"rgb{color[:3]}"),
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
    gt_ann_ids: Optional[set] = None,
    dt_ann_ids: Optional[set] = None,
    return_fig: bool = False,
) -> Optional[go.Figure]:
    """Display the image with the results.

    Args:
        cocoGt (COCO): Ground truth COCO object.
        cocoDt (COCO, optional): Detection COCO object. Default is None.
        image_id (int, optional): Image id to display. Default is 1.
        iouType (str, optional): Type of the annotation, one of 'bbox', 'segm', or 'keypoints'. Default is "bbox".
        display_fp (bool, optional): Display false positive annotations. Default is True.
        display_fn (bool, optional): Display false negative annotations. Default is True.
        display_tp (bool, optional): Display true positive annotations. Default is True.
        display_gt (bool, optional): Display ground truth annotations. Default is True.
        data_folder (str, optional): Folder containing the images. Default is None.
        categories (list, optional): List of category ids to display. Default is None.
        gt_ann_ids (set, optional): Set of ground truth annotation ids to display. Default is None.
        dt_ann_ids (set, optional): Set of detection annotation ids to display. Default is None.
        return_fig (bool, optional): Return the figure object instead of displaying it. Default is False.

    Returns:
        Optional[go.Figure]: The figure object if return_fig is True, otherwise None.
    """
    polygons = []

    image = cocoGt.imgs[image_id]
    gt_anns = {ann["id"]: ann for ann in cocoGt.imgToAnns[image_id]}

    if gt_ann_ids is not None:
        gt_anns = {id: ann for id, ann in gt_anns.items() if id in gt_ann_ids}

    if cocoDt is not None:
        dt_anns = {ann["id"]: ann for ann in cocoDt.imgToAnns[image_id]}

        if dt_ann_ids is not None:
            dt_anns = {id: ann for id, ann in dt_anns.items() if id in dt_ann_ids}

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
        logger.warning(f"[{image_load_path}] not found!\nLoading default empty image")

        im = Image.new("RGB", (image["width"], image["height"]))

    categories_labels = {category["id"]: category["name"] for _, category in cocoGt.cats.items()}
    category_id_to_skeleton = {category["id"]: category.get("skeleton") for _, category in cocoGt.cats.items()}

    if len(gt_anns) > 0:
        for ann in gt_anns.values():
            if categories is None or ann["category_id"] in categories:
                _text = [
                    "id={}".format(ann["id"]),
                    "category={}".format(categories_labels[ann["category_id"]]),
                ]
                if ann.get("fn", False):
                    if display_fn:
                        fn_text = ["<b>FN</b>"] + _text
                        poly = generate_ann_polygon(
                            ann,
                            color=FN_COLOR,
                            iouType=iouType,
                            text="<br>".join(fn_text),
                            legendgroup="fn",
                            category_id_to_skeleton=category_id_to_skeleton,
                        )
                        if poly is not None:
                            polygons.append(poly)
                else:
                    if display_gt:
                        gt_text = ["<b>GT</b>"] + _text
                        poly = generate_ann_polygon(
                            ann,
                            color=GT_COLOR,
                            iouType=iouType,
                            text="<br>".join(gt_text),
                            legendgroup="gt",
                            category_id_to_skeleton=category_id_to_skeleton,
                        )
                        if poly is not None:
                            polygons.append(poly)

    if len(dt_anns) > 0:
        for ann in dt_anns.values():
            if categories is None or ann["category_id"] in categories:
                _text = [
                    "id={}".format(ann["id"]),
                    "category={}".format(categories_labels[ann["category_id"]]),
                    "score={:.2f}".format(ann["score"]),
                ]
                if ann.get("tp", False):
                    if display_tp:
                        tp_text = ["<b>DT</b>"] + _text

                        if ann.get("mae") is not None:
                            tp_text.append("MAE={:.2f}".format(ann["mae"]))

                        if ann.get("iou") is not None:
                            tp_text.append("IoU={:.2f}".format(ann["iou"]))

                        poly = generate_ann_polygon(
                            ann,
                            color=DT_COLOR,
                            iouType=iouType,
                            text="<br>".join(tp_text),
                            legendgroup="tp",
                            category_id_to_skeleton=category_id_to_skeleton,
                        )
                        if poly is not None:
                            polygons.append(poly)
                else:
                    if display_fp:
                        fp_text = ["<b>FP</b>"] + _text

                        poly = generate_ann_polygon(
                            ann,
                            color=FP_COLOR,
                            iouType=iouType,
                            text="<br>".join(fp_text),
                            legendgroup="fp",
                            category_id_to_skeleton=category_id_to_skeleton,
                        )
                        if poly is not None:
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
        if poly is not None:
            if legends.get(poly.legendgroup) is None:
                poly.showlegend = True
                legends[poly.legendgroup] = True

        fig.add_trace(poly)

    layout = {
        "title": f"image_id={image_id}<br>image_fn={image_fn}",
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
) -> Optional[go.Figure]:
    """Display the confusion matrix.

    Args:
        conf_matrix (np.ndarray): Confusion matrix (shape: [n_classes, n_classes + 2]).
        labels (list): List of class labels.
        normalize (bool, optional): If True, normalize the confusion matrix to percentage. Default is False.
        return_fig (bool, optional): If True, return the figure object. Default is False.

    Returns:
        Optional[go.Figure]: The figure object if return_fig is True, otherwise None.
    """

    _labels = labels + ["fp", "fn"]

    if normalize:
        conf_matrix /= conf_matrix.sum(axis=1).reshape(-1, 1)
        conf_matrix *= 100

    hovertemplate = "Real: %{y}<br>Predict: %{x}<br>"

    if normalize:
        hovertemplate += "Percent: %{z:.0f}<extra></extra>"
    else:
        hovertemplate += "Count: %{z:.0f}<extra></extra>"

    annotations = []
    for j, row in enumerate(conf_matrix):
        annotations.append([])
        for value in row:
            text_value = f"{value:.0f}"
            if normalize:
                text_value += "%"

            annotations[j].append(text_value)

    heatmap = go.Heatmap(
        z=conf_matrix,
        x=_labels,
        y=_labels[:-2],
        text=annotations,
        colorscale="Blues",
        hovertemplate=hovertemplate,
        texttemplate="%{text}",
    )

    title = "Confusion Matrix"
    if normalize:
        title = "Normalized " + title

    layout = {
        "title": title,
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
    }

    fig = go.Figure(data=[heatmap], layout=layout)
    fig.update_traces(showscale=False)
    fig.update_layout(height=700, width=900)

    if return_fig:
        return fig

    fig.show()


def plot_pre_rec(curves, return_fig: bool = False):
    """Plot the precision-recall curve.

    Args:
        curves (list): List of curves to plot. Each element is a dict with keys: 'recall_list', 'precision_list', 'scores', 'name'.
        return_fig (bool, optional): If True, return the figure object. Default is False.

    Returns:
        Optional[go.Figure]: The figure object if return_fig is True, otherwise None.
    """  # noqa: E501
    fig = go.Figure()

    for _curve in curves:
        recall_list = _curve["recall_list"]
        precision_list = _curve["precision_list"]
        scores = _curve["scores"]

        if "name" in _curve and len(_curve["name"]) > 0:
            name = _curve["name"]
        elif "label" in _curve and len(_curve["label"]) > 0:
            name = _curve["label"]
        else:
            name = "Precision-Recall"

        fig.add_trace(
            go.Scatter(
                x=recall_list,
                y=precision_list,
                name=name,
                text=scores,
                hovertemplate="Pre: %{y:.3f}<br>" + "Rec: %{x:.3f}<br>" + "Score: %{text:.3f}<extra></extra>",
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
    """Plot the F1 confidence curve.

    Args:
        curves (list): List of curves to plot. Each element is a dict with keys: 'recall_list', 'precision_list', 'scores', 'label'.
        return_fig (bool, optional): If True, return the figure object. Default is False.

    Returns:
        Optional[go.Figure]: The figure object if return_fig is True, otherwise None.
    """  # noqa: E501
    fig = go.Figure()
    eps = 1e-16
    for _curve in curves:
        recall_list = _curve["recall_list"]
        precision_list = _curve["precision_list"][: len(recall_list)]
        scores = _curve["scores"]
        f1_curve = 2 * precision_list * recall_list / (precision_list + recall_list + eps)

        if "name" in _curve and len(_curve["name"]) > 0:
            name = _curve["name"]
        elif "label" in _curve and len(_curve["label"]) > 0:
            name = _curve["label"]
        else:
            name = "F1-Confidence"

        fig.add_trace(
            go.Scatter(
                x=scores,
                y=f1_curve,
                name=name,
                hovertemplate="F1: %{y:.3f}<br>" + "Confidence: %{x:.3f}<br><extra></extra>",
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


def plot_ced_metric(curves, normalize: bool = False, return_fig: bool = False):
    """Plot the Cumulative Error Distribution (CED) curve.

    Args:
        curves (list): List of ced curves to plot. Each dict must have keys: 'mae' (dict), 'category' (dict), optionally 'label'.
        normalize (bool, optional): If True, normalize values to percent. Default is False.
        return_fig (bool, optional): If True, return the figure object. Default is False.

    Returns:
        Optional[go.Figure]: The figure object if return_fig is True, otherwise None.
    """  # noqa: E501
    fig = go.Figure()

    if normalize:
        fig.layout.yaxis.title = "The proportion of the sample to the total sample [%]"
        _hovertemplate_y = "%{y:.2f}%<br>"
    else:
        fig.layout.yaxis.title = "Number of samples"
        _hovertemplate_y = "n=%{y}<br>"

    fig.layout.xaxis.title = "Mean Absolute error"

    traces = []
    for ced_curve in curves:
        for key, val in ced_curve["mae"].items():
            if normalize:
                y = (np.array(val["y"]) / max(val["y"])) * 100
            else:
                y = val["y"]

            category_name = ced_curve["category"]["name"]

            legendgrouptitle = f"CED Curve [{category_name}]"
            if ced_curve.get("label") is not None:
                legendgrouptitle = "[{}] ".format(ced_curve["label"]) + legendgrouptitle

            traces.append(
                go.Scatter(
                    x=val["x"],
                    y=y,
                    name=key,
                    hovertemplate=(
                        _hovertemplate_y + "mae: %{x:.2f}<br>" + f"{category_name} -> {key}<br>" + "<extra></extra>"
                    ),
                    showlegend=True,
                    mode="lines",
                    legendgroup=legendgrouptitle,
                    legendgrouptitle={
                        "text": legendgrouptitle,
                    },
                    visible=True if key == "MEAN" else False,  # "legendonly",
                )
            )

    fig.add_traces(traces)
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)

    updatemenus = [
        dict(
            type="dropdown",
            direction="down",
            y=1.1,
            x=1,
            buttons=list([
                dict(
                    args=[{"xaxis.type": "linear"}],
                    label="Linear Scale",
                    method="relayout",
                ),
                dict(
                    args=[{"xaxis.type": "log"}],
                    label="Log Scale",
                    method="relayout",
                ),
            ]),
        ),
        dict(
            type="dropdown",
            direction="down",
            y=1.1,
            x=0.85,
            buttons=list([
                dict(
                    args=[
                        {"visible": [x.name == "MEAN" for i, x in enumerate(traces)]},
                    ],
                    label="Display MEAN",
                    method="restyle",
                ),
                dict(
                    args=[
                        {"visible": [x.name != "MEAN" for i, x in enumerate(traces)]},
                    ],
                    label="Display ALL",
                    method="restyle",
                ),
            ]),
        ),
    ]

    layout = {
        "title": "Cumulative Error Distribution",
        "autosize": True,
        "height": 600,
        "width": 1200,
        # "xaxis_type" : "log",
        "updatemenus": updatemenus,
    }

    fig.update_layout(layout)

    if return_fig:
        return fig

    fig.show()


def show_anns(
    cocoGt: COCO,
    image_id: int,
    ann_ids: Optional[List[int]] = None,
    iouType: showAnnsiouTypeT = "bbox",
    data_folder: Optional[str] = None,
    return_fig: bool = False,
):
    """Show ground truth annotations on an image.

    Args:
        cocoGt (COCO): COCO object containing ground truth data.
        image_id (int): Image id to display.
        ann_ids (List[int], optional): List of annotation ids to show. Default is None (show all).
        iouType (str, optional): Type of the annotation, one of 'bbox' or 'segm'. Default is "bbox".
        data_folder (str, optional): Folder containing the images. Default is None.
        return_fig (bool, optional): Return the figure object instead of displaying it. Default is False.

    Returns:
        Optional[go.Figure]: The figure object if return_fig is True, otherwise None.
    """
    return display_image(
        cocoGt,
        image_id=image_id,
        iouType=iouType,
        display_gt=True,
        display_fn=False,
        display_fp=False,
        display_tp=False,
        data_folder=data_folder,
        gt_ann_ids=ann_ids,
        return_fig=return_fig,
    )
