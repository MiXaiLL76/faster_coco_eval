"""Regression coverage for low-risk core API correctness fixes."""

from pathlib import Path

import numpy as np
import pytest

from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.core.faster_eval_api import COCOeval


def _dataset() -> dict:
    """Build the smallest indexed detection dataset for core API checks."""
    return {
        "images": [{"id": 1, "height": 10, "width": 10}],
        "categories": [{"id": 1, "name": "object"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0.0, 0.0, 2.0, 2.0],
                "area": 4.0,
                "iscrowd": 0,
            }
        ],
    }


def test_coco_accepts_pathlib_paths_for_annotations_and_results(tmp_path: Path) -> None:
    """Path-like JSON inputs must work for construction and result loading."""
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        '{"images": [{"id": 1, "height": 10, "width": 10}], '
        '"categories": [{"id": 1, "name": "object"}], "annotations": []}'
    )
    results_path = tmp_path / "results.json"
    results_path.write_text('[{"image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2], "score": 1.0}]')

    coco = COCO(annotations_path)
    results = coco.loadRes(results_path)

    assert coco.dataset["images"][0]["id"] == 1
    assert results.getAnnIds() == [1]


@pytest.mark.parametrize("loader_name", ["loadAnns", "loadCats", "loadImgs"])
def test_coco_loaders_reject_non_integer_ids(loader_name: str) -> None:
    """Unsupported scalar or iterable IDs must fail with a useful type
    error."""
    coco = COCO(_dataset())

    with pytest.raises(TypeError, match="ids must"):
        getattr(coco, loader_name)(["not-an-integer"])


def test_to_dict_separate_fn_does_not_mutate_indexed_annotations() -> None:
    """Separating false negatives must only modify the exported dictionary."""
    coco = COCO(_dataset())
    coco.anns[1]["fn"] = True

    exported = coco.to_dict(separate_fn=True)

    assert coco.anns[1]["category_id"] == 1
    assert exported["annotations"][0]["category_id"] == 2
    assert len(coco.cats) == 1
    assert len(exported["categories"]) == 2


def test_python_auc_does_not_mutate_its_input_arrays() -> None:
    """Envelope computation must not alter precision or recall owned by
    callers."""
    recall = np.array([0.0, 0.5, 1.0])
    precision = np.array([1.0, 0.25, 0.5])

    COCOeval_faster.calc_auc(recall, precision, method="py")

    np.testing.assert_array_equal(recall, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(precision, [1.0, 0.25, 0.5])


def test_extra_metrics_require_matching_data_and_handle_empty_matches() -> None:
    """Extra metrics must reject premature calls and avoid empty-match
    division."""
    evaluator = COCOeval_faster(print_function=lambda *_: None)

    with pytest.raises(RuntimeError, match="Matching"):
        evaluator.compute_mIoU()
    with pytest.raises(RuntimeError, match="Accumulation"):
        evaluator.compute_mAUC()

    evaluator.eval = {"matched": {}}
    assert evaluator.compute_mIoU() == 0.0


def test_compatibility_evaluator_keeps_a_writable_print_function() -> None:
    """The pycocotools-compatible evaluator must retain its property setter."""
    evaluator = COCOeval()

    def sink(*_: object) -> None:
        """Discard evaluator output."""

    evaluator.print_function = sink

    assert evaluator.print_function is sink
