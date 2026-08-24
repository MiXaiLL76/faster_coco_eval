"""Regression coverage for C++ input-validation boundaries."""

import math
import pickle
from concurrent.futures import ThreadPoolExecutor
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


class TestDatasetConcurrency:
    """Locking behaviour of the native dataset cache."""

    @staticmethod
    def _populated_dataset(pairs: int = 64):
        """Build a dataset spanning several image/category pairs."""
        dataset = _eval.Dataset()
        for index in range(pairs):
            dataset.append_ref(index, 1, {"id": index, "area": 10.0, "iscrowd": 0})
            dataset.append_ref(index, 1, {"id": index + 1000, "area": 20.0, "iscrowd": 1})
        return dataset

    def test_concurrent_readers_complete(self):
        """Many threads reading the cache finish and agree on the contents.

        get_cpp_annotations populates a shared cache on miss, so the
        read path mutates. A recursive or mis-ordered lock here would
        deadlock and hang the suite rather than fail an assertion.
        """
        dataset = self._populated_dataset()
        results = []

        def read_all():
            results.append([len(dataset.get_cpp_annotations(i, 1)) for i in range(64)])

        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in [pool.submit(read_all) for _ in range(8)]:
                future.result(timeout=60)

        assert results == [[2] * 64] * 8

    def test_reads_interleaved_with_cache_eviction_complete(self):
        """Reads racing cache eviction return whole annotation lists.

        clear_cache_entry erases entries that get_cpp_annotations may be
        populating. Returning by value keeps callers safe from an entry
        being dropped mid-use; a reference would dangle instead.
        """
        dataset = self._populated_dataset()
        observed = []

        def read():
            observed.extend(len(dataset.get_cpp_annotations(i, 1)) for i in range(64))

        def evict():
            for i in range(64):
                dataset.clear_cache_entry(i, 1)

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(read if n % 2 == 0 else evict) for n in range(6)]
            for future in futures:
                future.result(timeout=60)

        assert set(observed) == {2}
