from unittest.mock import patch

import numpy as np
import pytest

from faster_coco_eval.extra.utils import (
    _check_opencv,
    conver_mask_to_poly,
    convert_ann_rle_to_poly,
    convert_rle_to_poly,
    opencv_available,
)


def test_check_opencv_available():
    """Test _check_opencv when OpenCV is available."""
    if opencv_available:
        # Should not raise an exception
        _check_opencv()
    else:
        # Should raise ImportError
        with pytest.raises(ImportError, match="Missing dependency: opencv-python"):
            _check_opencv()


@pytest.mark.skipif(not opencv_available, reason="OpenCV not available")
def test_conver_mask_to_poly_basic():
    """Test basic mask to polygon conversion."""
    # Create a simple mask with a rectangle
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 20:80] = 255
    bbox = [15, 25, 70, 50]

    polygons = conver_mask_to_poly(mask, bbox)

    # Should return at least one polygon
    assert isinstance(polygons, list)


@pytest.mark.skipif(not opencv_available, reason="OpenCV not available")
def test_conver_mask_to_poly_with_margin():
    """Test mask to polygon conversion with custom margin."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    bbox = [35, 35, 30, 30]

    polygons = conver_mask_to_poly(mask, bbox, boxes_margin=0.2)

    assert isinstance(polygons, list)


@pytest.mark.skipif(not opencv_available, reason="OpenCV not available")
def test_conver_mask_to_poly_boundary_cases():
    """Test mask to polygon conversion with boundary cases."""
    # Test with bbox extending beyond mask boundaries
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 255

    # Bbox that would extend beyond mask boundaries
    bbox = [0, 0, 60, 60]

    polygons = conver_mask_to_poly(mask, bbox)
    assert isinstance(polygons, list)

    # Test with negative coordinates after margin
    bbox = [2, 2, 10, 10]
    polygons = conver_mask_to_poly(mask, bbox, boxes_margin=0.5)
    assert isinstance(polygons, list)


@pytest.mark.skipif(not opencv_available, reason="OpenCV not available")
def test_conver_mask_to_poly_empty_mask():
    """Test mask to polygon conversion with empty mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    bbox = [10, 10, 50, 50]

    polygons = conver_mask_to_poly(mask, bbox)

    # Should return empty list for empty mask
    assert polygons == []


@pytest.mark.skipif(opencv_available, reason="OpenCV is available")
def test_conver_mask_to_poly_no_opencv():
    """Test that function raises error when OpenCV is not available."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    bbox = [10, 10, 50, 50]

    with pytest.raises(ImportError, match="Missing dependency: opencv-python"):
        conver_mask_to_poly(mask, bbox)


@patch("faster_coco_eval.extra.utils.mask_util.decode")
def test_convert_rle_to_poly(mock_decode):
    """Test RLE to polygon conversion."""
    # Mock the mask_util.decode function
    mock_mask = np.zeros((100, 100), dtype=np.uint8)
    mock_mask[30:70, 20:80] = 1
    mock_decode.return_value = mock_mask

    rle = {"size": [100, 100], "counts": b"some_rle_data"}
    bbox = [15, 25, 70, 50]

    if opencv_available:
        polygons = convert_rle_to_poly(rle, bbox)
        assert isinstance(polygons, list)
        mock_decode.assert_called_once_with(rle)
    else:
        with pytest.raises(ImportError):
            convert_rle_to_poly(rle, bbox)


def test_convert_ann_rle_to_poly_with_rle():
    """Test annotation conversion when segmentation is RLE."""
    ann = {"segmentation": {"size": [100, 100], "counts": b"some_rle_data"}, "bbox": [10, 10, 50, 50]}

    with patch("faster_coco_eval.extra.utils.convert_rle_to_poly") as mock_convert:
        mock_convert.return_value = [[10, 10, 60, 10, 60, 60, 10, 60]]

        result = convert_ann_rle_to_poly(ann)

        # Should store original RLE in 'counts'
        assert "counts" in result
        assert result["counts"] == {"size": [100, 100], "counts": b"some_rle_data"}

        # Should convert segmentation to polygon
        assert result["segmentation"] == [[10, 10, 60, 10, 60, 60, 10, 60]]

        mock_convert.assert_called_once_with({"size": [100, 100], "counts": b"some_rle_data"}, [10, 10, 50, 50])


def test_convert_ann_rle_to_poly_with_polygon():
    """Test annotation conversion when segmentation is already polygon."""
    ann = {"segmentation": [[10, 10, 60, 10, 60, 60, 10, 60]], "bbox": [10, 10, 50, 50]}

    result = convert_ann_rle_to_poly(ann)

    # Should not change polygon segmentation
    assert result["segmentation"] == [[10, 10, 60, 10, 60, 60, 10, 60]]

    # Should not add 'counts' field
    assert "counts" not in result


def test_convert_ann_rle_to_poly_preserves_other_fields():
    """Test that annotation conversion preserves other fields."""
    ann = {
        "segmentation": [[10, 10, 60, 10, 60, 60, 10, 60]],
        "bbox": [10, 10, 50, 50],
        "area": 2500,
        "id": 123,
        "category_id": 1,
    }

    result = convert_ann_rle_to_poly(ann)

    # Should preserve all other fields
    assert result["bbox"] == [10, 10, 50, 50]
    assert result["area"] == 2500
    assert result["id"] == 123
    assert result["category_id"] == 1
