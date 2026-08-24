from types import SimpleNamespace

import numpy as np

from faster_coco_eval.extra import draw


class FakeScatter:
    """Capture scatter arguments without requiring Plotly."""

    def __init__(self, **kwargs):
        """Store scatter arguments as attributes."""
        self.__dict__.update(kwargs)


class FakeFigure:
    """Capture plotted traces without requiring Plotly."""

    def __init__(self):
        """Initialize empty trace and axis collections."""
        self.data = []
        self.layout = SimpleNamespace(xaxis=SimpleNamespace(), yaxis=SimpleNamespace())

    def add_trace(self, trace):
        """Append one plotted trace."""
        self.data.append(trace)

    def add_traces(self, traces):
        """Append the plotted traces."""
        self.data.extend(traces)

    def update_layout(self, _layout):
        """Accept layout updates for compatibility with Plotly."""
        return None

    def update_xaxes(self, **_kwargs):
        """Accept x-axis updates for compatibility with Plotly."""
        return None

    def update_yaxes(self, **_kwargs):
        """Accept y-axis updates for compatibility with Plotly."""
        return None


def test_generate_ann_polygon_does_not_mutate_segmentation(monkeypatch):
    """Close polygon coordinates in a temporary list for repeatable
    rendering."""
    monkeypatch.setattr(draw, "go", SimpleNamespace(Scatter=FakeScatter))
    monkeypatch.setattr(draw, "plotly_available", True)
    segmentation = [[10, 10, 30, 10, 30, 30, 10, 30]]
    ann = {"segmentation": segmentation}

    first = draw.generate_ann_polygon(ann, (255, 0, 0, 0.5), iouType="segm")
    second = draw.generate_ann_polygon(ann, (255, 0, 0, 0.5), iouType="segm")

    assert segmentation == [[10, 10, 30, 10, 30, 30, 10, 30]]
    assert first.x == second.x
    assert first.y == second.y


def test_plot_curves_filter_invalid_precision(monkeypatch):
    """Do not plot the -1 precision sentinel or compute negative F1 values."""
    monkeypatch.setattr(draw, "go", SimpleNamespace(Figure=FakeFigure, Scatter=FakeScatter))
    monkeypatch.setattr(draw, "plotly_available", True)
    curves = [
        {
            "recall_list": np.array([0.0, 0.5, 1.0]),
            "precision_list": np.array([1.0, -1.0, 0.5]),
            "scores": np.array([0.9, 0.4, 0.1]),
            "label": "curve1",
        }
    ]

    precision_recall = draw.plot_pre_rec(curves, return_fig=True)
    f1_confidence = draw.plot_f1_confidence(curves, return_fig=True)

    np.testing.assert_array_equal(precision_recall.data[0].x, [0.0, 1.0])
    np.testing.assert_array_equal(precision_recall.data[0].y, [1.0, 0.5])
    np.testing.assert_array_equal(precision_recall.data[0].text, [0.9, 0.1])
    np.testing.assert_allclose(f1_confidence.data[0].x, [0.9, 0.1])
    np.testing.assert_allclose(f1_confidence.data[0].y, [0.0, 2 * 0.5 / 1.5])


def test_plot_ced_metric_normalizes_all_zero_curve_without_nan(monkeypatch):
    """Keep an empty CED series finite when normalization is requested."""
    monkeypatch.setattr(draw, "go", SimpleNamespace(Figure=FakeFigure, Scatter=FakeScatter))
    monkeypatch.setattr(draw, "plotly_available", True)
    curves = [{"mae": {"MEAN": {"x": [0.0, 1.0], "y": [0, 0]}}, "category": {"name": "cat"}}]

    figure = draw.plot_ced_metric(curves, normalize=True, return_fig=True)

    np.testing.assert_array_equal(figure.data[0].y, [0.0, 0.0])
