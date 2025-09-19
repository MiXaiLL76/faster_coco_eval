from unittest.mock import Mock, patch

import numpy as np
import pytest

from faster_coco_eval.extra.extra import ExtraEval


def test_extra_eval_init_assertions():
    """Test ExtraEval initialization assertions."""
    # Test assertion for None cocoGt
    with pytest.raises(AssertionError, match="cocoGt is empty"):
        ExtraEval(cocoGt=None)


def test_extra_eval_keypoints_usecats():
    """Test that keypoints iouType sets useCats to True."""
    mock_gt = Mock()
    mock_gt.anns = {}

    with patch.object(ExtraEval, "evaluate"):
        extra_eval = ExtraEval(
            cocoGt=mock_gt,
            cocoDt=None,
            iouType="keypoints",
            useCats=False,  # Should be overridden to True
            kpt_oks_sigmas=[0.026, 0.025, 0.025],
        )

        assert extra_eval.useCats is True
        assert extra_eval.kpt_oks_sigmas is not None
        assert isinstance(extra_eval.kpt_oks_sigmas, np.ndarray)


def test_extra_eval_non_keypoints():
    """Test ExtraEval for non-keypoints iouType."""
    mock_gt = Mock()
    mock_gt.anns = {}

    with patch.object(ExtraEval, "evaluate"):
        extra_eval = ExtraEval(cocoGt=mock_gt, cocoDt=None, iouType="bbox", useCats=True)

        assert extra_eval.useCats is True
        assert extra_eval.kpt_oks_sigmas is None


def test_evaluate_assertion():
    """Test evaluate method assertion for None cocoDt."""
    mock_gt = Mock()
    mock_gt.anns = {}

    # Don't patch evaluate to test the actual assertion
    extra_eval = ExtraEval(cocoGt=mock_gt, cocoDt=None)
    extra_eval.cocoDt = None

    with pytest.raises(AssertionError, match="cocoDt is empty"):
        extra_eval.evaluate()


def test_drop_cocodt_by_score_assertion():
    """Test drop_cocodt_by_score assertion for None cocoDt."""
    mock_gt = Mock()
    mock_gt.anns = {}

    with patch.object(ExtraEval, "evaluate"):
        extra_eval = ExtraEval(cocoGt=mock_gt, cocoDt=None)
        extra_eval.cocoDt = None

        with pytest.raises(AssertionError, match="cocoDt is empty"):
            extra_eval.drop_cocodt_by_score(0.5)


def test_drop_cocodt_by_score_zero():
    """Test drop_cocodt_by_score with zero min_score (should do nothing)."""
    mock_gt = Mock()
    mock_gt.anns = {}
    mock_dt = Mock()
    mock_dt.anns = {1: {"score": 0.3, "image_id": 1, "id": 1}}
    mock_dt.imgToAnns = {1: [{"score": 0.3, "image_id": 1, "id": 1}]}

    with patch.object(ExtraEval, "evaluate"):
        extra_eval = ExtraEval(cocoGt=mock_gt, cocoDt=None)
        extra_eval.cocoDt = mock_dt

        # Should not modify anything when min_score is 0
        extra_eval.drop_cocodt_by_score(0.0)
        assert len(mock_dt.anns) == 1


def test_fp_image_ann_map_empty():
    """Test fp_image_ann_map with no false positives."""
    mock_gt = Mock()
    mock_gt.anns = {}
    mock_dt = Mock()
    mock_dt.anns = {
        1: {"image_id": 1, "id": 1},  # No 'fp' key
        2: {"image_id": 2, "id": 2, "fp": False},  # fp is False
    }

    with patch.object(ExtraEval, "evaluate"):
        extra_eval = ExtraEval(cocoGt=mock_gt, cocoDt=None)
        extra_eval.cocoDt = mock_dt

        fp_map = extra_eval.fp_image_ann_map

        # Should be empty since no fp=True annotations
        assert len(fp_map) == 0


def test_fp_image_ann_map_with_fps():
    """Test fp_image_ann_map with false positives."""
    mock_gt = Mock()
    mock_gt.anns = {}
    mock_dt = Mock()
    mock_dt.anns = {
        1: {"image_id": 1, "id": 1, "fp": True},
        2: {"image_id": 1, "id": 2, "fp": True},
        3: {"image_id": 2, "id": 3, "fp": False},
        4: {"image_id": 2, "id": 4, "fp": True},
    }

    with patch.object(ExtraEval, "evaluate"):
        extra_eval = ExtraEval(cocoGt=mock_gt, cocoDt=None)
        extra_eval.cocoDt = mock_dt

        fp_map = extra_eval.fp_image_ann_map

        assert len(fp_map) == 2
        assert fp_map[1] == {1, 2}
        assert fp_map[2] == {4}


def test_fn_image_ann_map_empty():
    """Test fn_image_ann_map with no false negatives."""
    mock_gt = Mock()
    mock_gt.anns = {
        1: {"image_id": 1, "id": 1},  # No 'fn' key
        2: {"image_id": 2, "id": 2, "fn": False},  # fn is False
    }

    with patch.object(ExtraEval, "evaluate"):
        extra_eval = ExtraEval(cocoGt=mock_gt, cocoDt=None)

        fn_map = extra_eval.fn_image_ann_map

        # Should be empty since no fn=True annotations
        assert len(fn_map) == 0


def test_fn_image_ann_map_with_fns():
    """Test fn_image_ann_map with false negatives."""
    mock_gt = Mock()
    mock_gt.anns = {
        1: {"image_id": 1, "id": 1, "fn": True},
        2: {"image_id": 1, "id": 2, "fn": True},
        3: {"image_id": 2, "id": 3, "fn": False},
        4: {"image_id": 2, "id": 4, "fn": True},
    }

    with patch.object(ExtraEval, "evaluate"):
        extra_eval = ExtraEval(cocoGt=mock_gt, cocoDt=None)

        fn_map = extra_eval.fn_image_ann_map

        assert len(fn_map) == 2
        assert fn_map[1] == {1, 2}
        assert fn_map[2] == {4}
