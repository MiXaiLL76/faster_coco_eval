# Original work Copyright (c) Piotr Dollar and Tsung-Yi Lin, 2014.
# Modified work Copyright (c) 2024 MiXaiLL76
from typing import List, Union

import numpy as np

import faster_coco_eval.mask_api_new_cpp as _mask

ValidRleType = Union[
    List[np.ndarray], List[List[float]], np.ndarray, List[dict]
]


def iou(
    dt: ValidRleType,
    gt: ValidRleType,
    iscrowd: List[int],
):
    return _mask.iou(dt, gt, iscrowd)


def merge(rleObjs: List[dict], intersect: int = 0):
    """Merge a list of run-length encoded objects.

    Args:
        rleObjs (list of dict): run-length encoding of binary masks
        intersect (int): flag for type of merge to perform

    Returns:
        merged (dict): run-length encoding of merged mask

    """
    return _mask.merge(rleObjs, intersect)


def frPyObjects(
    objs: Union[
        ValidRleType,
        np.ndarray,
        List[float],
        dict,
    ],
    h: int,
    w: int,
) -> Union[dict, List[dict]]:
    """Convert a list of objects to a format suitable for use in the
    _mask.frPyObjects function.

    Args:
        objs (np.ndarray or list of list of float or dict):
            objects to be converted
        h (int): height of the image
        w (int): width of the image

    """
    return _mask.frPyObjects(objs, h, w)


def encode(bimask: np.ndarray) -> dict:
    """Encode binary masks using RLE.

    Args:
        bimask (ndarray): binary mask

    Returns:
        rle (dict): run-length encoding of the binary mask

    """
    if len(bimask.shape) == 3:
        return _mask.encode(bimask)
    elif len(bimask.shape) == 2:
        h, w = bimask.shape
        return _mask.encode(bimask.reshape((h, w, 1), order="F"))[0]


def decode(rleObjs: Union[dict, List[dict]]) -> np.ndarray:
    """Decode binary masks encoded via RLE.

    Args:
        rleObjs (dict or list of dict): run-length encoding of binary mask

    Returns:
        bimask (ndarray): decoded binary mask

    """
    if type(rleObjs) is list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:, :, 0]


def area(rleObjs: Union[dict, List[dict]]) -> np.ndarray:
    """
    Compute area of encoded masks.
    Args:
        rleObjs (dict or list of dict): run-length encoding of binary mask

    Returns:
        area (np.ndarray): area of run-length encodings
    """
    if type(rleObjs) is list:
        return _mask.area(rleObjs)
    else:
        return _mask.area([rleObjs])[0]


def toBbox(rleObjs: Union[dict, List[dict]]) -> np.ndarray:
    """Get bounding boxes surrounding encoded masks.

    Args:
        rleObjs (dict or list of dict): run-length encoding of binary mask
    Returns:
        bbox (np.ndarray): bounding box of run-length encodings

    """
    if type(rleObjs) is list:
        return _mask.toBbox(rleObjs)
    else:
        return _mask.toBbox([rleObjs])[0]
