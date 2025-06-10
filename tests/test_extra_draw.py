import os
import unittest

import numpy as np
import plotly.graph_objs as go
from PIL import Image

from faster_coco_eval.extra.draw import (
    display_image,
    display_matrix,
    generate_ann_polygon,
    plot_ced_metric,
    plot_f1_confidence,
    plot_pre_rec,
    show_anns,
)


class DummyCOCO:
    """Minimal COCO mock for tests."""

    def __init__(self):
        self.imgs = {1: {"file_name": "test.jpg", "width": 100, "height": 100}}
        self.imgToAnns = {1: [{"id": 1, "bbox": [10, 10, 20, 20], "category_id": 1}]}
        self.cats = {1: {"id": 1, "name": "class1", "skeleton": []}}


class TestExtraDraw(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.dummy_coco = DummyCOCO()
        # Create a fake jpg file for testing
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img.save("test.jpg")

    def tearDown(self):
        try:
            os.remove("test.jpg")
        except Exception:
            pass

    def test_generate_ann_polygon_bbox(self):
        ann = {"bbox": [10, 10, 20, 20], "category_id": 1}
        color = (255, 0, 0, 0.5)
        result = generate_ann_polygon(ann, color, iouType="bbox")
        self.assertIsInstance(result, go.Scatter)
        # Check x and y coordinates
        self.assertEqual(list(result.x), [10, 30, 30, 10, 10, None])
        self.assertEqual(list(result.y), [10, 10, 30, 30, 10, None])
        self.assertEqual(result.line["color"], "rgb(255, 0, 0)")
        self.assertEqual(result.fillcolor, "rgba(255, 0, 0, 0.5)")

    def test_generate_ann_polygon_segm(self):
        ann = {"bbox": [10, 10, 20, 20], "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]], "category_id": 1}
        color = (255, 0, 0, 0.5)
        # convert_ann_rle_to_poly just returns ann for this case (see import)
        result = generate_ann_polygon(ann, color, iouType="segm")
        self.assertIsInstance(result, go.Scatter)
        # Check that coordinates start with 10, 10
        self.assertTrue(result.x[0] == 10)
        self.assertTrue(result.y[0] == 10)
        self.assertEqual(result.line["color"], "rgb(255, 0, 0)")
        self.assertEqual(result.fillcolor, "rgba(255, 0, 0, 0.5)")

    def test_generate_ann_polygon_keypoints(self):
        ann = {
            "bbox": [10, 10, 20, 20],
            "keypoints": [20, 20, 2, 30, 30, 2, 10, 30, 2],
            "category_id": 1,
        }
        color = (255, 0, 0, 0.5)
        category_id_to_skeleton = {1: [(1, 2), (2, 3)]}
        result = generate_ann_polygon(
            ann,
            color,
            iouType="keypoints",
            category_id_to_skeleton=category_id_to_skeleton,
        )
        self.assertIsInstance(result, go.Scatter)
        # Contains closing bbox and 2 keypoint edges, but order may differ due to implementation details
        self.assertTrue(20 in result.x and 10 in result.y)

    def test_display_image(self):
        fig = display_image(self.dummy_coco, image_id=1, return_fig=True)
        self.assertIsInstance(fig, go.Figure)
        # Check that the title contains the correct image_id
        self.assertIn("image_id=1", fig.layout.title.text)
        # Check figure size
        self.assertEqual(fig.layout.height, 700)
        self.assertEqual(fig.layout.width, 900)

    def test_display_matrix(self):
        mat = np.array([[2, 1, 0], [1, 3, 0]])
        labels = ["class1"]
        fig = display_matrix(mat, labels, return_fig=True)
        self.assertIsInstance(fig, go.Figure)
        # Check correct labels
        self.assertEqual(list(fig.data[0].x), ["class1", "fp", "fn"])
        self.assertEqual(list(fig.data[0].y), ["class1"])
        # Check matrix values
        self.assertEqual(fig.data[0].z.tolist(), [[2, 1, 0], [1, 3, 0]])
        # Check for normalize
        mat_norm = np.array([[2, 1, 0], [1, 3, 0]], dtype=float)
        fig2 = display_matrix(mat_norm, labels, normalize=True, return_fig=True)
        self.assertIsInstance(fig2, go.Figure)
        # Check that at least one value is normalized (float, not int)
        self.assertTrue(any(isinstance(v, float) for v in fig2.data[0].z.flatten()))

    def test_plot_pre_rec(self):
        curves = [
            {
                "recall_list": [0, 0.5, 1.0],
                "precision_list": [1.0, 0.8, 0.6],
                "scores": [0.9, 0.6, 0.2],
                "name": "curve1",
            }
        ]
        fig = plot_pre_rec(curves, return_fig=True)
        self.assertIsInstance(fig, go.Figure)
        # Check that the plot has one line with the correct name
        self.assertEqual(fig.data[0].name, "curve1")
        # Check that x and y match the input
        np.testing.assert_array_equal(fig.data[0].x, [0, 0.5, 1.0])
        np.testing.assert_array_equal(fig.data[0].y, [1.0, 0.8, 0.6])

    def test_plot_f1_confidence(self):
        curves = [
            {
                "recall_list": np.array([0, 0.5, 1.0]),
                "precision_list": np.array([1.0, 0.8, 0.6]),
                "scores": np.array([0.9, 0.6, 0.2]),
                "label": "curve1",
            }
        ]
        fig = plot_f1_confidence(curves, return_fig=True)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.data[0].name, "curve1")
        # Check that F1 is calculated correctly (for first point F1==0, for second 2*0.8*0.5/(0.8+0.5)=0.615...)
        self.assertAlmostEqual(fig.data[0].y[1], 2 * 0.8 * 0.5 / (0.8 + 0.5), places=4)

    def test_plot_ced_metric(self):
        curves = [
            {
                "mae": {
                    "MEAN": {"x": [0.1, 0.2, 0.3], "y": [1, 2, 3]},
                    "OTHER": {"x": [0.1, 0.2, 0.3], "y": [2, 3, 4]},
                },
                "category": {"name": "test"},
                "label": "testlabel",
            }
        ]
        fig = plot_ced_metric(curves, return_fig=True)
        self.assertIsInstance(fig, go.Figure)
        # Check that the plot includes a trace named "MEAN"
        names = [trace.name for trace in fig.data]
        self.assertIn("MEAN", names)
        # Check that x and y in "MEAN" match input
        mean_trace = next(t for t in fig.data if t.name == "MEAN")
        np.testing.assert_array_equal(mean_trace.x, [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(mean_trace.y, [1, 2, 3])
        # Check normalization
        fig2 = plot_ced_metric(curves, normalize=True, return_fig=True)
        mean_trace2 = next(t for t in fig2.data if t.name == "MEAN")
        np.testing.assert_almost_equal(mean_trace2.y, [1 / 3 * 100, 2 / 3 * 100, 3 / 3 * 100], decimal=2)

    def test_show_anns(self):
        fig = show_anns(self.dummy_coco, image_id=1, return_fig=True)
        self.assertIsInstance(fig, go.Figure)
        self.assertIn("image_id=1", fig.layout.title.text)


if __name__ == "__main__":
    unittest.main()
