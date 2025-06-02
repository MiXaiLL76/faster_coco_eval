try:
    import cv2

    opencv_available = True
except ImportError:
    opencv_available = False

import numpy as np

import faster_coco_eval.core.mask as mask_util


def conver_mask_to_poly(mask: np.ndarray, bbox: list, boxes_margin: float = 0.1) -> list:
    """Convert a mask (uint8) to a list of polygons in COCO style.

    Args:
        mask (np.ndarray): The mask image as a numpy array.
        bbox (list): Bounding box of the annotation in the format [x, y, w, h].
        boxes_margin (float, optional): Margin factor to increase the bounding box size. Defaults to 0.1.

    Returns:
        list: List of polygons in COCO format (list of lists of coordinates).
    """
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h

    m_w, m_h = int(w * boxes_margin), int(h * boxes_margin)
    x1 -= m_w
    if x1 < 0:
        x1 = 0

    y1 -= m_h
    if y1 < 0:
        y1 = 0

    x2 += m_w
    y2 += m_h

    mask_h, mask_w = mask.shape[:2]

    if x2 > mask_w:
        x2 = mask_w

    if y2 > mask_h:
        y2 = mask_h

    coords, _ = cv2.findContours(
        mask[int(y1) : int(y2), int(x1) : int(x2)].astype(np.uint8),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    coords = list(coords)
    if len(coords) > 0:
        for cnt_idx, cnt in enumerate(coords):
            coords[cnt_idx] = cnt + [x1, y1]

        coords = [_cnt.ravel().tolist() for _cnt in coords if _cnt.shape[0] >= 6]

    return coords


def convert_rle_to_poly(rle: dict, bbox: list) -> list:
    """Convert RLE (Run-Length Encoding) to a list of polygons in COCO style.

    Args:
        rle (dict): RLE of the mask image (COCO format).
        bbox (list): Bounding box of the annotation in the format [x, y, w, h].

    Returns:
        list: List of polygons in COCO format (list of lists of coordinates).
    """
    mask = mask_util.decode(rle) * 255
    return conver_mask_to_poly(mask, bbox)


def convert_ann_rle_to_poly(ann: dict) -> dict:
    """Convert annotation segmentation from RLE to polygon style and save RLE
    in the 'counts' variable.

    Args:
        ann (dict): Annotation dictionary with at least 'segmentation' (RLE) and 'bbox' fields.

    Returns:
        dict: Annotation dictionary with 'segmentation' converted to polygons and original RLE stored in 'counts'.

    Raises:
        Exception: If OpenCV is not available and conversion is required.
    """
    if type(ann["segmentation"]) is dict:
        if not opencv_available:
            raise Exception("Your dataset needs to be converted to polygons. Install **opencv-python** for this.")

        ann["counts"] = ann["segmentation"]
        ann["segmentation"] = convert_rle_to_poly(ann["segmentation"], ann["bbox"])
    return ann
