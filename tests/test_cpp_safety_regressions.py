"""Regression coverage for C++ input-validation boundaries."""

import math
import pickle
from types import SimpleNamespace

import faster_coco_eval.faster_eval_api_cpp as _eval
import faster_coco_eval.mask_api_new_cpp as _mask
import numpy as np
import pytest

from faster_coco_eval import COCO, COCOeval_faster


class SerializedImageEvaluation(_eval.ImageEvaluation):
    """Create a native evaluation through its supported pickle state
    boundary."""

    def __init__(self, state):
        super().__init__()
        self._state = state

    def __getstate__(self):
        return self._state


def test_to_bbox_returns_zero_box_for_empty_foreground_rle():
    """An RLE with only a background run has an empty bounding box."""
    rle = _mask.frUncompressedRLE([{"size": [2, 2], "counts": [4]}])
    bbox = _mask.toBbox(rle)

    np.testing.assert_array_equal(bbox, [[0.0, 0.0, 0.0, 0.0]])


def test_rle_varint_rejects_more_than_thirteen_chunks():
    """Malformed compressed RLE cannot shift a varint past int64 width."""
    with pytest.raises(RuntimeError, match="varint exceeds 64-bit width"):
        _mask.RLE("P" * 13 + "0", 1, 1)


def test_evaluator_rejects_nan_detection_score_before_sorting():
    """NaN scores cannot enter the native strict-weak-order comparator."""
    coco_gt = COCO({
        "images": [{"id": 1, "height": 10, "width": 10}],
        "categories": [{"id": 1, "name": "object"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 2, 2],
                "area": 4,
                "iscrowd": 0,
            }
        ],
    })
    coco_dt = coco_gt.loadRes([{"image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2], "score": math.nan}])

    evaluator = COCOeval_faster(coco_gt, coco_dt, iouType="bbox")
    with pytest.raises(ValueError, match="Detection scores must not be NaN"):
        evaluator.evaluate()


@pytest.mark.parametrize(
    ("detection_matches", "detection_scores", "detection_ignores"),
    [
        pytest.param([0], [0.9], [False], id="non-rectangular-matches"),
        pytest.param([0] * 10, [0.9], [False], id="misaligned-ignore-buffer"),
        pytest.param([0] * 10, [0.9, 0.8], [False] * 10, id="misaligned-score-buffer"),
    ],
)
def test_accumulate_rejects_misaligned_result_buffers(detection_matches, detection_scores, detection_ignores):
    """Accumulation rejects result buffers that do not share one detection
    shape."""
    params = SimpleNamespace(
        recThrs=[0.0],
        maxDets=[1],
        iouThrs=[0.5] * 10,
        useCats=1,
        catIds=[1],
        areaRng=[[0, 100000]],
        imgIds=[1],
    )
    evaluation = pickle.loads(
        pickle.dumps(
            SerializedImageEvaluation((detection_matches, [0], detection_scores, [False], detection_ignores, []))
        )
    )

    with pytest.raises(RuntimeError, match="Detection result buffers must be rectangular"):
        _eval.COCOevalAccumulate(params, [evaluation])


def test_accumulate_rejects_nan_detection_score_before_sorting():
    """Accumulation rejects deserialized NaN scores before stable sorting."""
    params = SimpleNamespace(
        recThrs=[0.0],
        maxDets=[1],
        iouThrs=[0.5],
        useCats=1,
        catIds=[1],
        areaRng=[[0, 100000]],
        imgIds=[1],
    )
    evaluation = pickle.loads(pickle.dumps(SerializedImageEvaluation(([0], [0], [math.nan], [False], [False], []))))

    with pytest.raises(ValueError, match="Detection scores must not be NaN"):
        _eval.COCOevalAccumulate(params, [evaluation])


def _evaluate_with_ground_truth(annotation: dict) -> list[float]:
    """Run a one-image bbox evaluation over a single ground-truth annotation."""
    images = [{"id": 1, "width": 100, "height": 100}]
    categories = [{"id": 1, "name": "a"}]
    base = {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "area": 100.0}
    coco_gt = COCO({"images": images, "categories": categories, "annotations": [{**base, **annotation}]})
    coco_dt = coco_gt.loadRes([{"image_id": 1, "category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9}])

    evaluator = COCOeval_faster(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.maxDets = [10]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return list(evaluator.stats)


class TestAnnotationFieldParsing:
    """Field lookup in the native annotation parser."""

    @pytest.mark.parametrize(
        ("annotation", "expected_ap"),
        [
            pytest.param({"is_crowd": 1}, 1.0, id="is-crowd-only"),
            pytest.param({"iscrowd": 1}, -1.0, id="iscrowd-only"),
            pytest.param({"is_crowd": 0, "iscrowd": 1}, -1.0, id="both-spellings"),
            pytest.param({"is_crowd": 0}, 1.0, id="is-crowd-zero"),
            pytest.param({}, 1.0, id="neither"),
        ],
    )
    def test_crowd_spelling_outcomes_are_pinned(self, annotation, expected_ap):
        """Pin the observed outcome for every crowd-key spelling.

        The parser reads "is_crowd" and falls back to "iscrowd", but _prepare
        derives the ignore flag from "iscrowd" alone, so the two spellings are
        not interchangeable end to end. This characterizes that existing
        asymmetry so a change to how the parser looks keys up cannot shift it
        unnoticed; it is not an endorsement of the asymmetry.
        """
        assert _evaluate_with_ground_truth(annotation)[0] == pytest.approx(expected_ap)

    @pytest.mark.parametrize(
        "annotation",
        [
            pytest.param({"lvis_mark": "bad"}, id="unconvertible-lvis-mark"),
            pytest.param({"area": "bad"}, id="unconvertible-area"),
            pytest.param({"ignore": object()}, id="unconvertible-ignore"),
        ],
    )
    def test_unconvertible_field_falls_back_to_default(self, annotation):
        """A field that cannot be converted leaves its default and does not raise.

        Annotations come from user data, so the parser is deliberately
        best-effort per field. Propagating instead would turn a tolerated
        oddity in one optional field into a failed evaluation.
        """
        assert _evaluate_with_ground_truth(annotation) == _evaluate_with_ground_truth({})

    def test_absent_optional_fields_are_tolerated(self):
        """Omitting every optional field still evaluates.

        The base annotation carries no "score", "ignore", or "lvis_mark"; each
        must read as its default rather than as a missing-key error.
        """
        assert _evaluate_with_ground_truth({})[0] == pytest.approx(1.0)
