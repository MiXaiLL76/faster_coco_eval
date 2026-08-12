import inspect
import subprocess
import sys
import threading

import numpy as np
import pytest

from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.core.cocoeval import Params


def _make_eval(
    detection_bbox: list[float] | None = None,
    ranges: dict[str, list[float]] | None = None,
    iou_type: str = "bbox",
) -> COCOeval_faster:
    """Build a small evaluator for core regression tests."""
    gt = COCO()
    gt.dataset = {
        "images": [{"id": 1, "width": 20, "height": 20}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "segmentation": [[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0]],
                "area": 100.0,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "object"}],
    }
    gt.createIndex()
    dt = gt.loadRes([
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": detection_bbox or [0.0, 0.0, 10.0, 10.0],
            "segmentation": [[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0]],
            "score": 1.0,
        }
    ])
    evaluator_kwargs = {"print_function": lambda *_: None}
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


def test_evaluate_computes_iou_pairs_concurrently():
    """Independent image/category IoUs should overlap across worker threads."""

    class ObservableEvaluator(COCOeval_faster):
        """Record concurrent calls while retaining real IoU behavior."""

        def __init__(self, coco_gt: COCO, coco_dt: COCO):
            """Initialize synchronization state for two IoU workers."""
            super().__init__(coco_gt, coco_dt, iouType="segm", print_function=lambda *_: None)
            self.active_calls = 0
            self.max_active_calls = 0
            self.call_lock = threading.Lock()
            self.workers_ready = threading.Barrier(2)

        def computeIoU(self, imgId: int, catId: int) -> list[float] | np.ndarray:
            """Wait for a peer worker before computing the real IoU result."""
            with self.call_lock:
                self.active_calls += 1
                self.max_active_calls = max(self.max_active_calls, self.active_calls)
            try:
                self.workers_ready.wait(timeout=1)
                return super().computeIoU(imgId, catId)
            finally:
                with self.call_lock:
                    self.active_calls -= 1

    base_evaluator = _make_eval(iou_type="segm")
    evaluator = ObservableEvaluator(base_evaluator.cocoGt, base_evaluator.cocoDt)
    evaluator.params.imgIds = [1, 2]
    evaluator.evaluate()

    assert (evaluator.max_active_calls, list(evaluator.ious)) == (2, [(1, 1), (2, 1)])


def test_load_res_accepts_empty_results():
    """An empty result set should produce an indexed empty COCO result."""
    evaluator = _make_eval()
    empty = evaluator.cocoGt.loadRes([])

    assert empty.dataset["annotations"] == []
    assert empty.dataset["categories"] == evaluator.cocoGt.dataset["categories"]
    assert empty.getAnnIds() == []
    assert empty.loadAnns([]) == []


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
