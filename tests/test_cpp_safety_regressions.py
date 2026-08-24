"""Regression coverage for C++ input-validation boundaries."""

import math
import multiprocessing
import pickle
import threading
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


def _populated_dataset(pairs: int = 64):
    """Build a dataset spanning several image/category pairs."""
    dataset = _eval.Dataset()
    for index in range(pairs):
        dataset.append_ref(index, 1, {"id": index, "area": 10.0, "iscrowd": 0})
        dataset.append_ref(index, 1, {"id": index + 1000, "area": 20.0, "iscrowd": 1})
    return dataset


def _exercise_concurrent_readers():
    """Synchronize readers before they populate the shared native cache."""
    dataset = _populated_dataset()
    barrier = threading.Barrier(8)
    results = []

    def read_all():
        barrier.wait()
        results.append([len(dataset.get_cpp_annotations(i, 1)) for i in range(64)])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: read_all(), range(8)))

    assert results == [[2] * 64] * 8


def _exercise_reads_with_cache_eviction():
    """Synchronize cache readers and eviction workers before contention."""
    dataset = _populated_dataset()
    barrier = threading.Barrier(6)
    observed = []

    def read():
        barrier.wait()
        observed.extend(len(dataset.get_cpp_annotations(i, 1)) for i in range(64))

    def evict():
        barrier.wait()
        for index in range(64):
            dataset.clear_cache_entry(index, 1)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(read if index % 2 == 0 else evict) for index in range(6)]
        for future in futures:
            future.result()

    assert set(observed) == {2}


def _exercise_clean_finalizer_reentry():
    """Re-enter Dataset while clean releases its annotation references."""
    dataset = _eval.Dataset()

    class ReenterDataset(dict):
        def __del__(self):
            len(dataset)

    dataset.append_ref(1, 1, ReenterDataset(id=1, area=10.0, iscrowd=0))
    dataset.clean()
    assert len(dataset) == 0


def _exercise_load_finalizer_reentry():
    """Re-enter Dataset while load_tuple replaces existing annotations."""
    dataset = _eval.Dataset()

    class ReenterDataset(dict):
        def __del__(self):
            len(dataset)

    dataset.append_ref(1, 1, ReenterDataset(id=1, area=10.0, iscrowd=0))
    dataset.load_tuple((0, []))
    assert len(dataset) == 0


def _exercise_load_conversion_reentry():
    """Re-enter Dataset through a numeric conversion in load_tuple."""
    dataset = _eval.Dataset()

    class ReenterFloat:
        def __float__(self):
            len(dataset)
            return 1.0

    dataset.load_tuple((1, [(ReenterFloat(), 1.0, [{"id": 1}])]))
    assert len(dataset) == 1


def _exercise_parse_conversion_reentry():
    """Re-enter Dataset through annotation conversion on a cache miss."""
    dataset = _eval.Dataset()

    class ReenterFloat:
        def __float__(self):
            len(dataset)
            return 10.0

    dataset.append_ref(1, 1, {"id": 1, "area": ReenterFloat(), "iscrowd": 0})
    assert len(dataset.get_cpp_annotations(1, 1)) == 1


def _exercise_writer_during_conversion():
    """Prevent a converted stale snapshot from replacing a newer cache."""
    dataset = _eval.Dataset()
    conversion_started = threading.Event()
    append_completed = threading.Event()

    class WaitForAppend:
        def __float__(self):
            conversion_started.set()
            append_completed.wait()
            return 10.0

    dataset.append_ref(1, 1, {"id": 1, "area": WaitForAppend(), "iscrowd": 0})

    def append_annotation():
        conversion_started.wait()
        dataset.append_ref(1, 1, {"id": 2, "area": 20.0, "iscrowd": 0})
        append_completed.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        append_future = pool.submit(append_annotation)
        assert len(dataset.get_cpp_annotations(1, 1)) == 2
        append_future.result()


def _assert_completes_in_subprocess(target):
    """Fail externally when a deadlock-sensitive scenario does not finish."""
    process = multiprocessing.get_context("spawn").Process(target=target)
    process.start()
    process.join(timeout=15)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail(f"{target.__name__} did not complete within 15 seconds")
    assert process.exitcode == 0


class TestDatasetConcurrency:
    """Locking behaviour of the native dataset cache."""

    def test_concurrent_readers_complete(self):
        """Synchronized cache readers finish within an external timeout."""
        _assert_completes_in_subprocess(_exercise_concurrent_readers)

    def test_reads_interleaved_with_cache_eviction_complete(self):
        """Synchronized readers and evictors return whole cache entries."""
        _assert_completes_in_subprocess(_exercise_reads_with_cache_eviction)

    @pytest.mark.parametrize(
        "scenario",
        [
            pytest.param(_exercise_clean_finalizer_reentry, id="clean-finalizer"),
            pytest.param(_exercise_load_finalizer_reentry, id="load-finalizer"),
            pytest.param(_exercise_load_conversion_reentry, id="load-conversion"),
            pytest.param(_exercise_parse_conversion_reentry, id="parse-conversion"),
        ],
    )
    def test_python_reentry_completes(self, scenario):
        """Python callbacks cannot self-deadlock on the Dataset mutex."""
        _assert_completes_in_subprocess(scenario)

    def test_writer_during_conversion_does_not_publish_stale_cache(self):
        """A cache miss retries when a writer changes its source snapshot."""
        _assert_completes_in_subprocess(_exercise_writer_during_conversion)

    def test_append_invalidates_parsed_cache(self):
        """Native reads include annotations appended after cache population."""
        dataset = _eval.Dataset()
        dataset.append_ref(1, 1, {"id": 1, "area": 10.0, "iscrowd": 0})
        assert len(dataset.get_cpp_annotations(1, 1)) == 1

        dataset.append_ref(1, 1, {"id": 2, "area": 20.0, "iscrowd": 0})

        assert [annotation["id"] for annotation in dataset.get(1, 1)] == [1, 2]
        assert len(dataset.get_cpp_annotations(1, 1)) == 2

    def test_dataset_pickle_round_trip_preserves_python_and_native_data(self):
        """Dataset pickle exercises __setstate__ and preserves all contents."""
        dataset = _eval.Dataset()
        annotations = [
            {"id": 1, "area": 10.0, "iscrowd": 0},
            {"id": 2, "area": 20.0, "iscrowd": 1},
        ]
        for annotation in annotations:
            dataset.append_ref(1, 1, annotation)
        dataset.append_ref(2, 1, {"id": 3, "area": 30.0, "iscrowd": 0})

        restored = pickle.loads(pickle.dumps(dataset))

        assert len(restored) == 2
        assert restored.get(1, 1) == annotations
        assert [len(restored.get_cpp_annotations(image_id, 1)) for image_id in (1, 2)] == [2, 1]
