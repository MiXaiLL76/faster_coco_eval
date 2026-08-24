from types import SimpleNamespace

import numpy as np

from faster_coco_eval.extra.curves import Curves


def test_build_curve_uses_real_category_ids_and_masks_sentinel_values(caplog):
    """Build curves with real ids and omit categories with only invalid
    precision."""
    curves = Curves.__new__(Curves)
    curves.cocoGt = SimpleNamespace(cats={1: {"name": "cat"}, 17: {"name": "dog"}})
    curves.eval = {
        "precision": np.array([[[[[1.0]], [[-1.0]]], [[[0.5]], [[-1.0]]], [[[0.25]], [[-1.0]]]]]),
        "scores": np.array([[[[[0.9]], [[-1.0]]], [[[0.5]], [[-1.0]]], [[[0.1]], [[-1.0]]]]]),
    }
    curves.recThrs = np.array([0.0, 0.5, 1.0])
    curves.useCats = True

    with caplog.at_level("WARNING"):
        result = curves.build_curve("category_id")

    assert len(result) == 1
    assert result[0]["category_id"] == 1
    assert result[0]["label"] == "[category_id=1] "
    np.testing.assert_array_equal(result[0]["recall_list"], [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(result[0]["precision_list"], [1.0, 0.5, 0.25])
    np.testing.assert_array_equal(result[0]["scores"], [0.9, 0.5, 0.1])
    assert "17" in caplog.text


def _build_ced_curves(mae_values):
    """Build CED curves from one-keypoint annotations with known MAEs."""
    curves = Curves.__new__(Curves)
    curves.cocoGt = SimpleNamespace(
        cats={1: {"id": 1, "name": "cat", "keypoints": ["nose"]}},
        anns={index: {"keypoints": [0.0, 0.0, 2], "matched": True, "dt_id": index} for index in range(len(mae_values))},
        get_ann_ids=lambda cat_ids: list(range(len(mae_values))),
    )
    curves.cocoDt = SimpleNamespace(anns={index: {"keypoints": [mae, mae, 2]} for index, mae in enumerate(mae_values)})
    curves.eval = {}
    curves.iouType = "keypoints"
    return curves.build_ced_curve(mae_count=3)


def test_build_ced_curve_keeps_x_values_nondecreasing():
    """Keep CED sample points ordered through the final endpoint."""
    curves = _build_ced_curves([0.0, 1.0, 2.0, 3.0])

    x_values = curves[0]["mae"]["MEAN"]["x"]
    assert np.all(np.diff(x_values) >= 0)


def test_build_ced_curve_avoids_repeating_zero_error_points():
    """Represent an all-zero error distribution with its endpoints only."""
    curves = _build_ced_curves([0.0])

    assert curves[0]["mae"]["MEAN"] == {"x": [0, 0.0], "y": [0, 1]}
