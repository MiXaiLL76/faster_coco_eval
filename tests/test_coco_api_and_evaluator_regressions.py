import inspect
import os
import subprocess
import sys
import threading
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor

import numpy as np
import pytest

import faster_coco_eval.core.faster_eval_api as faster_eval_api_module
from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.core.cocoeval import Params


def _make_eval(
    detection_bbox: list[float] | None = None,
    ranges: dict[str, list[float]] | None = None,
    iou_type: str = "bbox",
    include_second_pair: bool = False,
    num_pairs: int | None = None,
    rle_iou_max_workers: int | None = None,
) -> COCOeval_faster:
    """Build a small evaluator for core regression tests."""
    images = [{"id": 1, "width": 20, "height": 20}]
    annotations = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "segmentation": [[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0]],
            "area": 100.0,
            "iscrowd": 0,
        }
    ]
    detections = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": detection_bbox or [0.0, 0.0, 10.0, 10.0],
            "segmentation": [[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0]],
            "score": 1.0,
        }
    ]
    pair_count = num_pairs if num_pairs is not None else (2 if include_second_pair else 1)
    for image_id in range(2, pair_count + 1):
        images.append({"id": image_id, "width": 20, "height": 20})
        annotations.append({
            "id": image_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "segmentation": [[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0]],
            "area": 100.0,
            "iscrowd": 0,
        })
        detections.append({
            "image_id": image_id,
            "category_id": 1,
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "segmentation": [[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0]],
            "score": 1.0,
        })

    gt = COCO()
    gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }
    gt.createIndex()
    dt = gt.loadRes(detections)
    evaluator_kwargs = {"print_function": lambda *_: None}
    if rle_iou_max_workers is not None:
        evaluator_kwargs["rle_iou_max_workers"] = rle_iou_max_workers
    if ranges is not None:
        evaluator_kwargs["ranges"] = ranges
    return COCOeval_faster(gt, dt, iouType=iou_type, **evaluator_kwargs)


def test_invalid_core_iou_types_raise_value_error():
    """Reject forged invalid IoU types at each defensive core-Python guard."""
    evaluator = _make_eval()
    evaluator.params.iouType = "invalid"
    evaluator._prepare()
    with pytest.raises(ValueError, match="iouType"):
        evaluator.computeIoU(1, 1)

    evaluator.eval = {"ready": True}
    with pytest.raises(ValueError, match="iouType"):
        evaluator.summarize()

    evaluator.all_stats = np.array([])
    with pytest.raises(ValueError, match="iouType"):
        _ = evaluator.stats_as_dict


def test_math_matches_clears_annotations_before_a_second_run():
    """Re-evaluation must classify unmatched annotations from the current
    run."""
    evaluator = _make_eval(detection_bbox=[0.0, 0.0, 8.0, 8.0])
    evaluator.extra_calc = True

    evaluator.params.iouThrs = np.array([0.5])
    evaluator.evaluate()
    evaluator.accumulate()
    assert evaluator.cocoDt.anns[1]["tp"] is True

    evaluator.params.iouThrs = np.array([0.75])
    evaluator.evaluate()
    evaluator.accumulate()

    assert evaluator.cocoDt.anns[1].get("tp") is None
    assert evaluator.cocoDt.anns[1]["fp"] is True
    assert evaluator.cocoGt.anns[1].get("dt_id") is None
    assert evaluator.cocoGt.anns[1]["fn"] is True


@pytest.mark.parametrize(
    ("use_categories", "expected_keys", "expected_calls"),
    [
        pytest.param(
            True,
            {(1, 1), (1, 2), (2, 1), (2, 2)},
            [(1, 1)],
            id="category-aware",
        ),
        pytest.param(
            False,
            {(1, -1), (2, -1)},
            [(1, -1)],
            id="merged-categories",
        ),
    ],
)
def test_sparse_iou_dispatch_preserves_empty_public_pairs(
    use_categories: bool,
    expected_keys: set[tuple[int, int]],
    expected_calls: list[tuple[int, int]],
):
    """Only GT/DT intersections compute IoU while public keys stay complete."""

    class CountingEvaluator(COCOeval_faster):
        """Record IoU calls while retaining the evaluator implementation."""

        def __init__(self, coco_gt: COCO, coco_dt: COCO):
            """Initialize the evaluator and its observed call list."""
            super().__init__(coco_gt, coco_dt, iouType="bbox", print_function=lambda *_: None)
            self.iou_calls: list[tuple[int, int]] = []

        def computeIoU(self, imgId: int, catId: int) -> list[float] | np.ndarray:
            """Record the pair before running the real bbox calculation."""
            self.iou_calls.append((imgId, catId))
            return super().computeIoU(imgId, catId)

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": [
            {"id": 1, "width": 20, "height": 20},
            {"id": 2, "width": 20, "height": 20},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100},
        ],
        "categories": [{"id": 1, "name": "one"}, {"id": 2, "name": "two"}],
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes([
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 1.0},
        {"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.5},
    ])
    evaluator = CountingEvaluator(coco_gt, coco_dt)
    evaluator.params.useCats = int(use_categories)
    evaluator.evaluate()

    assert set(evaluator.ious) == expected_keys
    assert evaluator.iou_calls == expected_calls
    assert evaluator.ious[(2, 1 if use_categories else -1)] == []


def test_accumulation_is_deterministic_across_category_area_tasks():
    """Repeated native accumulation must preserve every result tensor."""
    evaluator = _make_eval(include_second_pair=True)

    evaluator.evaluate()
    evaluator.accumulate()
    first_result = {key: np.array(evaluator.eval[key], copy=True) for key in ("precision", "recall", "scores")}
    first_counts = list(evaluator.eval["counts"])
    first_matches = dict(evaluator.eval["matched"])

    evaluator.evaluate()
    evaluator.accumulate()

    for key, expected in first_result.items():
        np.testing.assert_array_equal(evaluator.eval[key], expected)
    assert evaluator.eval["counts"] == first_counts
    assert evaluator.eval["matched"] == first_matches


def test_evaluate_rle_iou_worker_cap_two_overlaps_real_iou_results(monkeypatch: pytest.MonkeyPatch):
    """Two RLE IoUs should overlap and retain the serial evaluator's values."""
    expected_concurrent_pairs = 2

    class ObservableEvaluator(COCOeval_faster):
        """Record concurrent calls while retaining real IoU behavior."""

        def __init__(self, coco_gt: COCO, coco_dt: COCO):
            """Initialize synchronization state for two IoU workers."""
            super().__init__(
                coco_gt,
                coco_dt,
                iouType="segm",
                print_function=lambda *_: None,
                rle_iou_max_workers=expected_concurrent_pairs,
            )
            self.active_calls = 0
            self.max_active_calls = 0
            self.call_lock = threading.Lock()
            self.workers_ready = threading.Barrier(expected_concurrent_pairs)

        def computeIoU(self, imgId: int, catId: int) -> list[float] | np.ndarray:
            """Wait for a peer worker before computing the real IoU result."""
            with self.call_lock:
                self.active_calls += 1
                self.max_active_calls = max(self.max_active_calls, self.active_calls)
            try:
                self.workers_ready.wait(timeout=2)
                return super().computeIoU(imgId, catId)
            finally:
                with self.call_lock:
                    self.active_calls -= 1

    monkeypatch.setattr(os, "process_cpu_count", lambda: expected_concurrent_pairs, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 1)

    serial_evaluator = _make_eval(
        iou_type="segm",
        include_second_pair=True,
        rle_iou_max_workers=1,
    )
    serial_evaluator.evaluate()

    base_evaluator = _make_eval(iou_type="segm", include_second_pair=True)
    evaluator = ObservableEvaluator(base_evaluator.cocoGt, base_evaluator.cocoDt)
    evaluator.params.imgIds = [1, 2]
    evaluator.evaluate()

    assert evaluator.max_active_calls == 2
    assert set(evaluator.ious) == {(1, 1), (2, 1)}
    for pair, expected_iou in serial_evaluator.ious.items():
        np.testing.assert_allclose(evaluator.ious[pair], expected_iou)


def test_constructor_exposes_rle_iou_worker_cap_default():
    """The public evaluator should expose its bounded RLE worker default."""
    parameter = inspect.signature(COCOeval_faster).parameters["rle_iou_max_workers"]

    evaluator = _make_eval(iou_type="segm")

    assert parameter.default == 8
    assert evaluator.rle_iou_max_workers == 8


@pytest.mark.parametrize("invalid_value", [True, 1.5, "2"], ids=["bool", "float", "string"])
def test_constructor_rejects_non_integer_rle_iou_worker_caps(invalid_value: object):
    """Worker-cap validation must reject values that cannot safely size a
    pool."""
    with pytest.raises(ValueError, match="rle_iou_max_workers"):
        _make_eval(rle_iou_max_workers=invalid_value)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_value", [0, -1], ids=["zero", "negative"])
def test_constructor_rejects_non_positive_rle_iou_worker_caps(invalid_value: int):
    """Worker caps must remain positive so scheduling has a valid serial
    floor."""
    with pytest.raises(ValueError, match="rle_iou_max_workers.*positive"):
        _make_eval(rle_iou_max_workers=invalid_value)


def test_evaluate_rle_iou_worker_cap_one_stays_serial(monkeypatch: pytest.MonkeyPatch):
    """A one-worker RLE evaluation must avoid a pool and retain its real
    IoU."""

    def unexpected_executor(*_: object, **__: object) -> object:
        pytest.fail("rle_iou_max_workers=1 must use the serial path")

    monkeypatch.setattr(faster_eval_api_module, "ThreadPoolExecutor", unexpected_executor)
    evaluator = _make_eval(iou_type="segm", rle_iou_max_workers=1)

    evaluator.evaluate()

    assert evaluator.params.compute_rle is True
    np.testing.assert_allclose(evaluator.ious[(1, 1)], np.array([[1.0]]))


@pytest.mark.parametrize(
    ("process_cpu_count", "affinity", "cpu_count", "configured_cap", "pair_count", "expected_workers"),
    [
        pytest.param(3, {0, 1}, 4, 8, 9, 3, id="process_capacity"),
        pytest.param(16, {0}, 1, 2, 9, 2, id="configured_cap"),
        pytest.param(16, {0}, 1, 8, 3, 3, id="pair_count"),
        pytest.param(16, {0}, 1, None, 9, 8, id="public_default_cap"),
        pytest.param(None, {0, 1}, 4, 8, 9, 2, id="affinity_fallback"),
        pytest.param(None, None, 4, 8, 9, 4, id="host_count_fallback"),
    ],
)
def test_evaluate_limits_rle_iou_workers_by_capacity_cap_and_pair_count(
    monkeypatch: pytest.MonkeyPatch,
    process_cpu_count: int | None,
    affinity: set[int] | None,
    cpu_count: int,
    configured_cap: int | None,
    pair_count: int,
    expected_workers: int,
):
    """The effective RLE pool must honor capacity fallbacks and public
    limits."""
    observed_worker_counts: list[int] = []

    class ImmediateExecutor:
        """Observe stdlib executor construction while completing real IoUs."""

        def __init__(self, max_workers: int):
            """Store the effective worker count requested by the evaluator."""
            observed_worker_counts.append(max_workers)

        def __enter__(self) -> "ImmediateExecutor":
            """Return the executor as required by the context-manager API."""
            return self

        def __exit__(self, *_: object) -> None:
            """Provide the stdlib executor context-manager exit contract."""

        def submit(self, function: object, *args: object) -> Future[object]:
            """Run the submitted IoU synchronously and return its future
            result."""
            future: Future[object] = Future()
            future.set_result(function(*args))  # type: ignore[operator]
            return future

    monkeypatch.setattr(os, "process_cpu_count", lambda: process_cpu_count, raising=False)
    if affinity is None:
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    else:
        monkeypatch.setattr(os, "sched_getaffinity", lambda _: affinity, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
    monkeypatch.setattr(faster_eval_api_module, "ThreadPoolExecutor", ImmediateExecutor)
    evaluator = _make_eval(
        iou_type="segm",
        num_pairs=pair_count,
        rle_iou_max_workers=configured_cap,
    )

    evaluator.evaluate()

    assert observed_worker_counts == [expected_workers]


def test_evaluate_bounds_pending_rle_iou_submission_while_workers_block(monkeypatch: pytest.MonkeyPatch):
    """Blocked workers must not make evaluation queue every image/category
    pair."""
    effective_workers = 2
    pair_count = 10

    class BlockingEvaluator(COCOeval_faster):
        """Block real IoU computations after worker threads begin executing."""

        def __init__(self, coco_gt: COCO, coco_dt: COCO):
            """Initialize the real evaluator plus deterministic worker
            signals."""
            super().__init__(
                coco_gt,
                coco_dt,
                iouType="segm",
                print_function=lambda *_: None,
                rle_iou_max_workers=effective_workers,
            )
            self.started_calls = 0
            self.calls_lock = threading.Lock()
            self.workers_started = threading.Event()
            self.release_workers = threading.Event()

        def computeIoU(self, imgId: int, catId: int) -> list[float] | np.ndarray:
            """Block each running worker before returning its real RLE IoU."""
            with self.calls_lock:
                self.started_calls += 1
                if self.started_calls == effective_workers:
                    self.workers_started.set()
            assert self.release_workers.wait(timeout=5), "test did not release blocked IoU workers"
            return super().computeIoU(imgId, catId)

    class ObservedExecutor:
        """Wrap the stdlib executor to expose submission count to the test."""

        submitted_count = 0

        def __init__(self, max_workers: int):
            """Create a real executor that runs the evaluator's real IoUs."""
            self.executor = RealThreadPoolExecutor(max_workers=max_workers)

        def __enter__(self) -> "ObservedExecutor":
            """Return the executor as required by the context-manager API."""
            return self

        def __exit__(self, *_: object) -> None:
            """Wait for released workers just like the stdlib executor does."""
            self.executor.shutdown(wait=True)

        def submit(self, function: object, *args: object) -> Future[object]:
            """Count every task accepted by the stdlib executor boundary."""
            type(self).submitted_count += 1
            return self.executor.submit(function, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "process_cpu_count", lambda: effective_workers, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    monkeypatch.setattr(faster_eval_api_module, "ThreadPoolExecutor", ObservedExecutor)
    base_evaluator = _make_eval(iou_type="segm", num_pairs=pair_count)
    evaluator = BlockingEvaluator(base_evaluator.cocoGt, base_evaluator.cocoDt)
    evaluator.params.imgIds = list(range(1, pair_count + 1))
    evaluation_errors: list[BaseException] = []

    def evaluate() -> None:
        """Capture an evaluation-thread exception for the main test
        assertion."""
        try:
            evaluator.evaluate()
        except BaseException as error:
            evaluation_errors.append(error)

    evaluation_thread = threading.Thread(target=evaluate)
    evaluation_thread.start()
    try:
        assert evaluator.workers_started.wait(timeout=2), "two RLE IoU workers did not start"
        assert ObservedExecutor.submitted_count <= 2 * effective_workers
        assert ObservedExecutor.submitted_count < pair_count
    finally:
        evaluator.release_workers.set()
        evaluation_thread.join(timeout=5)

    assert not evaluation_thread.is_alive()
    assert evaluation_errors == []


def test_load_res_accepts_empty_results():
    """An empty result set should produce an indexed empty COCO result."""
    evaluator = _make_eval()
    empty = evaluator.cocoGt.loadRes([])

    assert empty.dataset["annotations"] == []
    assert empty.dataset["categories"] == evaluator.cocoGt.dataset["categories"]
    assert empty.getAnnIds() == []
    assert empty.loadAnns([]) == []


def test_load_res_rejects_unsupported_result_types():
    """Unsupported result inputs must fail with the documented TypeError."""
    evaluator = _make_eval()

    with pytest.raises(TypeError, match="is not supported"):
        evaluator.cocoGt.loadRes(1)


def test_show_anns_rejects_unsupported_annotation_types():
    """Annotations without an instance or caption payload must be rejected."""
    coco = COCO()

    with pytest.raises(Exception, match="datasetType not supported"):
        coco.showAnns([{"id": 1}])


def test_dump_round_trips_indexed_dataset(tmp_path):
    """Dumped datasets must load with the same indexed COCO content."""
    evaluator = _make_eval()
    output_file = tmp_path / "dataset.json"

    evaluator.cocoGt.dump(output_file)
    loaded = COCO(output_file)

    assert loaded.dataset == evaluator.cocoGt.to_dict()
    assert loaded.getImgIds() == [1]
    assert loaded.getAnnIds() == [1]
    assert loaded.getCatIds() == [1]


def test_core_collection_defaults_are_not_shared_mutable_objects():
    """Core public collection parameters use None sentinels instead of mutable
    defaults."""
    methods = [
        COCO.getAnnIds,
        COCO.getCatIds,
        COCO.getImgIds,
        COCO.loadAnns,
        COCO.loadCats,
        COCO.loadImgs,
        COCO.download,
        COCO.get_ann_ids,
        COCO.get_cat_ids,
        COCO.get_img_ids,
        COCOeval_faster.__init__,
        Params.__init__,
    ]
    for method in methods:
        defaults = [
            parameter.default
            for parameter in inspect.signature(method).parameters.values()
            if parameter.default is not inspect.Parameter.empty
        ]
        assert not any(isinstance(default, (list, dict, set)) for default in defaults), method


def test_cocoeval_import_handles_unknown_cpu_count():
    """Importing the evaluator must tolerate platforms without a CPU count."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; os.cpu_count = lambda: None; import faster_coco_eval.core.cocoeval",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_custom_summary_layout_matches_configured_ranges_and_max_dets():
    """Summary labels and windows must match custom range and max-detection
    layouts."""
    ranges = {
        "tiny": [0, 16],
        "small": [16, 32],
        "medium": [32, 96],
        "large": [96, 1e5**2],
    }
    evaluator = _make_eval(ranges=ranges)
    evaluator.params.maxDets = [10]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    expected_labels = [
        "AP_all",
        "AP_50",
        "AP_75",
        "AP_tiny",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_10",
        "AR_tiny",
        "AR_small",
        "AR_medium",
        "AR_large",
        "AR_50",
        "AR_75",
    ]
    assert list(evaluator.stats_as_dict) == expected_labels
    assert len(evaluator.all_stats) == len(expected_labels)
    assert len(evaluator.stats) == 3 + len(ranges) * 2 + len(evaluator.params.maxDets)


class TestGetAnns:
    """Direct annotation lookup used by COCOeval._prepare."""

    @staticmethod
    def _dataset() -> COCO:
        """Two images, two categories, one annotation each combination."""
        return COCO({
            "images": [{"id": 1, "width": 20, "height": 20}, {"id": 2, "width": 20, "height": 20}],
            "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
            "annotations": [
                {"id": 10, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5], "area": 25.0, "iscrowd": 0},
                {"id": 11, "image_id": 1, "category_id": 2, "bbox": [1, 1, 5, 5], "area": 25.0, "iscrowd": 0},
                {"id": 12, "image_id": 2, "category_id": 1, "bbox": [2, 2, 5, 5], "area": 25.0, "iscrowd": 0},
            ],
        })

    @pytest.mark.parametrize(
        ("img_ids", "cat_ids"),
        [
            pytest.param([1, 2], [1, 2], id="all-images-all-categories"),
            pytest.param([1], [1, 2], id="one-image"),
            pytest.param([1, 2], [1], id="one-category"),
            pytest.param([], [], id="no-filter"),
            pytest.param([2], [2], id="empty-intersection"),
        ],
    )
    def test_matches_load_anns_of_get_ann_ids(self, img_ids, cat_ids):
        """getAnns returns exactly what loadAnns(getAnnIds(...)) returns.

        _prepare swapped the id round trip for this call, so any divergence in
        selection or ordering would silently change which detections land in
        each image/category bucket and shift the evaluation result.
        """
        coco = self._dataset()

        direct = coco.getAnns(imgIds=img_ids, catIds=cat_ids)
        via_ids = coco.loadAnns(coco.getAnnIds(imgIds=img_ids, catIds=cat_ids))

        assert [ann["id"] for ann in direct] == [ann["id"] for ann in via_ids]

    def test_returns_the_indexed_objects_not_copies(self):
        """Returned dicts are the index's own objects.

        _prepare mutates annotations in place (the ``ignore`` flag) and expects
        those writes to be visible to the evaluator afterwards.
        """
        coco = self._dataset()

        anns = coco.getAnns(imgIds=[1], catIds=[1])

        assert anns[0] is coco.anns[10]

    def test_unknown_image_id_does_not_grow_the_index(self):
        """Querying a missing image id leaves img_ann_map untouched.

        img_ann_map is a defaultdict, so a bare subscript would insert an empty
        list for every unknown id and slowly leak entries across evaluations.
        """
        coco = self._dataset()
        before = set(coco.img_ann_map)

        assert coco.getAnns(imgIds=[999], catIds=[1]) == []
        assert set(coco.img_ann_map) == before


class TestDatasetAppendBatch:
    """Batched annotation ingress used by COCOeval._prepare."""

    @staticmethod
    def _ann(drop=..., image_id: int = 1, category_id: int = 1) -> dict:
        """One annotation, optionally carrying a ``drop`` marker."""
        ann = {"image_id": image_id, "category_id": category_id, "id": 1}
        if drop is not ...:
            ann["drop"] = drop
        return ann

    @pytest.mark.parametrize(
        "drop",
        [
            pytest.param(..., id="key-absent"),
            pytest.param(False, id="false"),
            pytest.param(True, id="true"),
            pytest.param(0, id="int-zero"),
            pytest.param(1, id="int-one"),
            pytest.param(None, id="none"),
            pytest.param("", id="empty-string"),
            pytest.param("yes", id="non-empty-string"),
            pytest.param([], id="empty-list"),
            pytest.param([0], id="non-empty-list"),
        ],
    )
    def test_skip_dropped_follows_python_truthiness(self, drop):
        """A dropped annotation is decided by truthiness, not a bool cast.

        _prepare replaced `if not dt.get("drop", False)` with this native call.
        Annotations come from user-supplied data, so "drop" can hold any object;
        a bool cast raises on a string where the original loop just evaluated
        it, silently turning valid input into a crash.
        """
        from faster_coco_eval.faster_eval_api_cpp import Dataset

        ann = self._ann(drop)
        expected_kept = not ann.get("drop", False)

        dataset = Dataset()
        dataset.append_batch([ann], True)

        assert (len(dataset) > 0) == expected_kept

    def test_returns_encountered_image_category_pairs(self):
        """The returned set replaces the caller's own pair-collecting loop.

        _prepare intersects the ground-truth and detection pair sets to decide
        which IoU pairs are non-empty, so a wrong set silently changes which
        image/category combinations get evaluated at all.
        """
        from faster_coco_eval.faster_eval_api_cpp import Dataset

        anns = [self._ann(image_id=1, category_id=1), self._ann(image_id=2, category_id=3)]

        pairs = Dataset().append_batch(anns, False)

        assert pairs == {(1, 1), (2, 3)}

    def test_dropped_annotations_are_absent_from_returned_pairs(self):
        """A dropped annotation contributes neither storage nor a pair.

        If a dropped detection still registered its pair, _prepare would treat
        that image/category as non-empty and evaluate a bucket with no
        detections in it.
        """
        from faster_coco_eval.faster_eval_api_cpp import Dataset

        anns = [self._ann(image_id=1, category_id=1), self._ann(True, image_id=2, category_id=3)]

        pairs = Dataset().append_batch(anns, True)

        assert pairs == {(1, 1)}
