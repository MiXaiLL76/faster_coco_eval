try:
    import cv2

    opencv_available = True
except ImportError:
    opencv_available = False

import numpy as np

import faster_coco_eval.core.mask as mask_util


def conver_mask_to_poly(mask: np.ndarray, bbox: list, boxes_margin=0.1) -> list:
    """Convert mask (uint8) to list of poly as coco style.

    :param mask (np.ndarray): numpy array in uint8 0-255
    :param bbox (list): x,y,w,h of this ann
    :param boxes_margin (float): margin for increase bbox
    :return: list of poly as coco style

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

    coords, _ = cv2.findContours(
        mask[int(y1) : int(y2), int(x1) : int(x2)].astype(np.uint8),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    coords = list(coords)
    if len(coords) > 0:
        for cnt_idx, cnt in enumerate(coords):
            coords[cnt_idx] = cnt + [x1, y1]

        coords = [
            _cnt.ravel().tolist() for _cnt in coords if _cnt.shape[0] >= 6
        ]

    return coords


def convert_rle_to_poly(rle: dict, bbox: list):
    """Convert rle (dict) to list of poly as coco style.

    :param rle (dict): rle of mask image
    :param bbox (list): x,y,w,h of this ann
    :return: list of poly as coco style

    """
    mask = mask_util.decode(rle) * 255
    return conver_mask_to_poly(mask, bbox)


def convert_ann_rle_to_poly(ann: dict):
    """Convert ann segm from rle to poly style; Save rle in *count* var;

    :param ann (dict): ann row
    :return: ann

    """
    if type(ann["segmentation"]) is dict:
        if not opencv_available:
            raise Exception(
                "Your dataset needs to be converted to polygons. Install"
                " **opencv-python** for this."
            )

        ann["counts"] = ann["segmentation"]
        ann["segmentation"] = convert_rle_to_poly(
            ann["segmentation"], ann["bbox"]
        )
    return ann
