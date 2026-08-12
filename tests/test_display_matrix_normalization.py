from types import SimpleNamespace

import numpy as np

from faster_coco_eval.extra import draw


class FakeHeatmap:
    """Capture heatmap parameters without requiring Plotly."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeFigure:
    """Provide the Plotly figure methods used by the display helper."""

    def __init__(self, data=None, layout=None, **kwargs):
        self.data = data
        self.layout = layout

    def update_traces(self, **kwargs):
        self.trace_updates = kwargs

    def update_layout(self, **kwargs):
        self.layout_updates = kwargs


def test_display_matrix_normalization_preserves_input_and_handles_empty_rows(monkeypatch):
    """Normalize a copy and map rows with no samples to zero percentages."""
    fake_go = SimpleNamespace(Figure=FakeFigure, Heatmap=FakeHeatmap)
    monkeypatch.setattr(draw, "go", fake_go)
    monkeypatch.setattr(draw, "plotly_available", True)
    matrix = np.array([[2.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    original_matrix = matrix.copy()

    figure = draw.display_matrix(matrix, ["class1", "class2"], normalize=True, return_fig=True)

    np.testing.assert_array_equal(matrix, original_matrix)
    np.testing.assert_allclose(figure.data[0].z[0], [40.0, 20.0, 20.0, 20.0])
    assert np.isfinite(figure.data[0].z).all()
    np.testing.assert_array_equal(figure.data[0].z[1], [0.0, 0.0, 0.0, 0.0])
