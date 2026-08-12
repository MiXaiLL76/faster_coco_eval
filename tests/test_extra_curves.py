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
